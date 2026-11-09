"""Partial time smoothing of Task 1 predictions inside runs of consecutive target cells (outside the blanks).

  python t1_run_smooth.py --inp t1pred_x --out t1pred_xr [--av 0.25] [--aq 0.25] [--w 1]

Task 3 scores the conservation residual of N = q / v * length along time with the truth on every non-target cell, so
the physics cost of a link is the total variation of the error N_sub - N_true: independent jitter between two
consecutive predicted slots adds to it. Each predicted value becomes (1 - a) * own + a * (mean over the cell and its
run neighbours within +-w slots). Cells of the panel-wide blanks (t1_blank_mix.blank_cells) keep their values and
break the runs (they are smoothed by t1_blank_smooth.py).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import WORK, PANELS
from t1_blank_mix import blank_cells
from t1_blank_smooth import smooth_runs


def run_smooth(A: np.ndarray, blank: np.ndarray, w: int, a: float) -> np.ndarray:
    if a <= 0:
        return A
    R = np.where(blank, np.nan, A)
    S = smooth_runs(R, w)
    return np.where(np.isfinite(R), (1 - a) * A + a * S, A).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=1)
    ap.add_argument("--av", type=float, default=0.25, help="smoothing weight for speed")
    ap.add_argument("--aq", type=float, default=0.25, help="smoothing weight for flow")
    a = ap.parse_args()
    os.makedirs(f"{WORK}/{a.out}", exist_ok=True)
    for p in PANELS:
        for s in ("validation", "private"):
            z = np.load(f"{WORK}/{a.inp}/{p}_{s}.npz")
            b = blank_cells(p, s)
            np.savez_compressed(f"{WORK}/{a.out}/{p}_{s}.npz", V=run_smooth(z["V"], b, a.w, a.av),
                                Q=run_smooth(z["Q"], b, a.w, a.aq))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
