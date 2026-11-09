"""Recent queue activity per link from strictly earlier days (masked layer, causal).

The masked layers of train, validation and private are stitched into one calendar
(2030-06-01 .. 2031-04-30). For a window on day D at slot t, the feature for a link is
the share of observed cells that were queued in the previous `days` days within
+-`half` slots of t (and of t+6). Only days before D are read.
"""
from __future__ import annotations

import os

import numpy as np

from common import WORK, load, network

DAYS = 7
HALF = 6


def queued_calendar(panel: str):
    """(N_days, 288, L) int8: 1 queued, 0 free, -1 unobserved, over the stitched calendar."""
    path = f"{WORK}/t2/qcal_{panel}.npz"
    from common import CACHE
    # the splits cached for this panel (a newly generated month may come as one scored split only)
    present = [s for s in ("train", "validation", "private") if os.path.exists(f"{CACHE}/{panel}_{s}.npz")]
    if os.path.exists(path):
        z = np.load(path)
        # reuse only while it matches the cached splits (a rebuilt or new scored month changes their dates)
        now = [str(x) for s in present
               for x in np.load(f"{CACHE}/{panel}_{s}.npz", allow_pickle=True)["dates"]]
        if [str(x) for x in z["dates"]] == now:
            return z["Q"], list(z["dates"])
    net = network(panel)
    vcut = net.vcut.to_numpy(np.float32)
    parts, dates = [], []
    for s in present:
        z = load(panel, s)
        v = z["mv"]; el = z["melig"].astype(bool)
        q = np.where(np.isfinite(v) & el, (v <= vcut).astype(np.int8), np.int8(-1))
        parts.append(q.astype(np.int8)); dates += [str(x) for x in z["dates"]]
        del z
    Q = np.concatenate(parts)
    np.savez(path, Q=Q, dates=np.array(dates))
    return Q, dates


def recent_features(Q: np.ndarray, day: int, slot: int, days: int = DAYS, half: int = HALF):
    lo = max(0, day - days)
    if lo >= day:
        L = Q.shape[2]
        return np.full(L, np.nan, np.float32), np.full(L, np.nan, np.float32), np.full(L, np.nan, np.float32)
    out = []
    for c in (slot, min(slot + 6, 287)):
        a, b = max(0, c - half), min(288, c + half + 1)
        blk = Q[lo:day, a:b]
        obs = (blk >= 0).sum(axis=(0, 1)); qd = (blk == 1).sum(axis=(0, 1))
        out.append(np.where(obs > 0, qd / np.maximum(obs, 1), np.nan).astype(np.float32))
    # any queue on the link on each earlier day near this slot (day-level frequency)
    a, b = max(0, slot - half), min(288, slot + 6 + half + 1)
    blk = Q[lo:day, a:b]
    dayany = (blk == 1).any(axis=1).mean(axis=0).astype(np.float32)
    return out[0], out[1], dayany
