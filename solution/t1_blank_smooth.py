"""Centred moving average along time of a blank-cell prediction folder (values on the blank cells only).

  python t1_blank_smooth.py --inp t1pred_blank_v3 --out t1pred_blank_v3s [--w 1] [--flow]

Each run of consecutive predicted slots on a link is smoothed on its own (the window shrinks at the run's edges), so
the observed cells around a blank are never touched. Speed only unless --flow.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import WORK, PANELS


def smooth_runs(A: np.ndarray, w: int) -> np.ndarray:
    """A (D, T, L) with NaN outside the runs: mean over the finite cells within +-w slots of the same run."""
    fin = np.isfinite(A)
    X = np.where(fin, A, 0.0).astype(np.float64)
    S = X.copy(); N = fin.astype(np.float64)
    T = A.shape[1]
    for k in range(1, w + 1):
        for sh in (k, -k):
            # a neighbour counts only if every slot between it and the cell is finite (same run)
            ok = np.ones_like(fin)
            for j in range(1, k + 1):
                ok &= np.roll(fin, sh // k * j, axis=1)
            if sh > 0:
                ok[:, :sh] = False
            else:
                ok[:, T + sh:] = False
            ok &= fin
            S += np.where(ok, np.roll(X, sh, axis=1), 0.0); N += ok
    return np.where(fin, S / np.maximum(N, 1), np.nan).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=1)
    ap.add_argument("--flow", action="store_true")
    a = ap.parse_args()
    os.makedirs(f"{WORK}/{a.out}", exist_ok=True)
    for p in PANELS:
        for s in ("validation", "private"):
            z = np.load(f"{WORK}/{a.inp}/{p}_{s}.npz")
            V = smooth_runs(z["V"], a.w)
            Q = smooth_runs(z["Q"], a.w) if a.flow else z["Q"]
            np.savez_compressed(f"{WORK}/{a.out}/{p}_{s}.npz", V=V, Q=Q)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
