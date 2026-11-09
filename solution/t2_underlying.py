"""Estimate the underlying (complete, noise-reduced) queue state from observations.

The organizer labels queues from the simulator's underlying state, which is complete
and noise-free. Observations miss ~24% of cells and carry ~1.7 km/h of noise, so the
observed queue set flickers. Unobserved cells are filled by linear interpolation in
time between the nearest finite observations of the same link; cells whose speed lies
within `band` km/h of the cutoff take the median of their three-step neighbourhood.
"""
from __future__ import annotations

import numpy as np


def fill_time(v: np.ndarray, max_gap: int = 12) -> np.ndarray:
    """v: (N, L). Linear interpolation along axis 0 over gaps up to max_gap steps."""
    N, L = v.shape
    out = v.copy()
    fin = np.isfinite(v)
    pos = np.arange(N)[:, None]
    prev = np.maximum.accumulate(np.where(fin, pos, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(fin, pos, N)[::-1], axis=0)[::-1]
    cols = np.broadcast_to(np.arange(L), (N, L))
    hp = prev >= 0; hn = nxt < N
    pv = np.where(hp, v[np.clip(prev, 0, N - 1), cols], np.nan)
    nv = np.where(hn, v[np.clip(nxt, 0, N - 1), cols], np.nan)
    gp = pos - prev; gn = nxt - pos
    w = gp / np.maximum(gp + gn, 1)
    interp = np.where(hp & hn, pv + (nv - pv) * w, np.where(hp, pv, nv))
    ok = ~fin & ((hp & (gp <= max_gap)) | (hn & (gn <= max_gap)))
    out[ok] = interp[ok]
    return out


def underlying_queue(v: np.ndarray, vcut: np.ndarray, band: float = 3.0) -> np.ndarray:
    """v: (N, L) observed speeds (NaN where missing). Returns bool (N, L) queue estimate."""
    f = fill_time(v)
    q = f <= vcut[None, :]
    amb = np.abs(f - vcut[None, :]) < band
    if amb.any():
        pad = np.vstack([f[:1], f, f[-1:]])
        med = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
        q = np.where(amb, med <= vcut[None, :], q)
    return q & np.isfinite(f)
