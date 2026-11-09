"""Task 2: what the window selector says about an ongoing window, from data before T only.

The selector keeps the earliest qualifying origin. If the origin 5 minutes earlier was already
allowed by the 360-minute spacing (from published window origins before T) and already showed a
queue in its visible history with enough coverage, it can only have been skipped because its
persistence IoU was above 0.9: the queue stayed the same from T-5 through T+25. On train windows
built with the reverse-engineered selector this holds for 63 % of ongoing windows, where repeating
the state scores 0.951 IoU over T+5..T+25 and only 0.406 at T+30.

The flag uses: the window's own and earlier window origins (window_index), the released history
(T-60..T-5) and the masked layer at T-65. Nothing stamped after T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import RELEASE, load, network
from t2_data import HIST

SPACING_MIN = 360


def stable_flags(panel: str, split: str) -> dict[str, bool]:
    z = load(panel, split)
    vcut = network(panel).vcut.to_numpy(np.float32)
    dates = [str(x) for x in z["dates"]]
    links = [str(x) for x in z["links"]]
    li = {l: k for k, l in enumerate(links)}
    L = len(links)
    wi = pd.read_csv(f"{RELEASE}/task2/{panel}/{split}/window_index.csv")
    wi["o"] = pd.to_datetime(wi.forecast_origin, utc=True)
    hist = pd.read_parquet(f"{RELEASE}/task2/{panel}/{split}/window_history.parquet")
    hist["link_id"] = hist.link_id.astype(str)
    t0 = pd.Timestamp(wi.date.min() + "T00:00:00Z") + pd.Timedelta(minutes=5 * HIST)
    out = {}
    for w in wi.itertuples():
        if w.condition != "queue_ongoing":
            out[w.window_id] = False
            continue
        earlier = wi[wi.o < w.o]
        allowed = max([x + pd.Timedelta(minutes=SPACING_MIN) for x in earlier.o], default=t0)
        slack = (w.o - allowed).total_seconds() / 300
        h = hist[hist.window_id == w.window_id]
        k = ((w.o - pd.to_datetime(h.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
        assert (k >= 1).all() and (k <= HIST).all()
        c = h.link_id.map(li).to_numpy()
        V = np.full((HIST + 1, L), np.nan, np.float32); EL = np.zeros((HIST + 1, L), bool)
        V[HIST + 1 - k, c] = h.speed_kmh.to_numpy(np.float32)          # row 12 = T-5 ... row 1 = T-60
        EL[HIST + 1 - k, c] = h.is_score_eligible.to_numpy().astype(bool)
        d = dates.index(w.date); t = w.o.hour * 12 + w.o.minute // 5
        if t - HIST - 1 >= 0:                                           # row 0 = T-65 from the masked layer
            V[0] = z["mv"][d, t - HIST - 1]; EL[0] = z["melig"][d, t - HIST - 1].astype(bool)
        prev = slice(0, HIST)                                           # origin T-5 sees T-65..T-10
        Q = np.isfinite(V[prev]) & EL[prev] & (V[prev] <= vcut)
        pl = Q.sum(0)
        cov = (np.isfinite(V[prev]) & EL[prev]).sum() / (HIST * L)
        out[w.window_id] = bool(slack >= 1 and pl.sum() >= 2 and pl.max() >= 2 and cov >= 0.7)
    return out


SEL_NAMES = ["sel_stable1", "sel_fresh1", "sel_run", "sel_vis_age"]
MAXBACK = 72


def selection_features(Q: np.ndarray, F: np.ndarray, slack: float) -> np.ndarray:
    """What the selector's skipped origins imply, for one ongoing window.

    Q, F: (n, L) queued / finite-eligible observations for the rows before T, oldest first, last row = T-5
    (rows inside the released history raw, older rows from the masked layer). An origin T-5j was allowed
    if j <= slack; it saw rows [-12-j+1 .. -j] relative to T-5.
    Returns, per window: stable1 (T-5 skipped for persistence), fresh1 (T-5 had no visible queue),
    run (consecutive skipped-for-persistence origins), vis_age (consecutive origins with a visible queue)."""
    n, L = Q.shape
    cq = np.concatenate([np.zeros((1, L), np.int32), np.cumsum(Q, axis=0, dtype=np.int32)])
    cf = np.concatenate([np.zeros((1, F.shape[1]), np.int32), np.cumsum(F, axis=0, dtype=np.int32)]).sum(1)

    def ok(j):          # origin T-5j sees T-5j-60 .. T-5j-5 = rows n-j-12 .. n-j-1 (row n-1 is T-5)
        hi = n - j; lo = hi - HIST
        if lo < 0:
            return None
        pl = cq[hi] - cq[lo]
        return pl.sum() >= 2 and pl.max() >= 2, (cf[hi] - cf[lo]) / (HIST * L) >= 0.7
    r1 = ok(1)
    stable1 = bool(slack >= 1 and r1 is not None and r1[0] and r1[1])
    fresh1 = bool(slack >= 1 and r1 is not None and not r1[0])
    run = 0
    while run < min(slack, MAXBACK):
        r = ok(run + 1)
        if r is None or not (r[0] and r[1]):
            break
        run += 1
    vis = 0
    while vis < MAXBACK:
        r = ok(vis + 1)
        if r is None or not r[0]:
            break
        vis += 1
    return np.array([stable1, fresh1, run, vis], np.float32)
