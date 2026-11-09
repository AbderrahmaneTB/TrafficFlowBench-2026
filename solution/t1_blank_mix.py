"""Task 1 predictions with the cells of the panel-wide blanks taken from another source.

The March / April masked layers hide every link of a queue panel for 90 minutes after each official Task 2 origin
(T+5 ... T+90). Those cells are Task 1 targets (~2 % of all targets) but no training row looks like them: every
neighbour in time and space is missing and the month low-rank fits fall back to the link's monthly mean there.

  python t1_blank_mix.py --base t1pred_ens --blank t1pred_lg1 [--w 1] --out t1pred_mix

A target cell is "blank" when fewer than 5 % of the panel's links are observed in its (day, slot).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import WORK, PANELS, load


def blank_cells(panel: str, split: str) -> np.ndarray:
    """(D, 288, L) bool: slots where the released layer shows < 5 % of the panel's links."""
    z = load(panel, split)
    frac = np.isfinite(z["mv"]).mean(axis=2)
    return np.broadcast_to((frac < 0.05)[:, :, None], z["mv"].shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--blank", required=True, help="prediction folder used inside the blanks")
    ap.add_argument("--w", type=float, default=1.0, help="weight of --blank inside the blanks")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(f"{WORK}/{a.out}", exist_ok=True)
    n = 0
    for p in PANELS:
        for s in ("validation", "private"):
            b = np.load(f"{WORK}/{a.base}/{p}_{s}.npz"); x = np.load(f"{WORK}/{a.blank}/{p}_{s}.npz")
            m = blank_cells(p, s) & np.isfinite(b["V"]) & np.isfinite(x["V"]) & np.isfinite(x["Q"])
            V = b["V"].copy(); Q = b["Q"].copy()
            V[m] = (1 - a.w) * b["V"][m] + a.w * x["V"][m]
            Q[m] = (1 - a.w) * b["Q"][m] + a.w * x["Q"][m]
            n += int(m.sum())
            np.savez_compressed(f"{WORK}/{a.out}/{p}_{s}.npz", V=V.astype(np.float32), Q=Q.astype(np.float32))
    print("wrote", a.out, "blank cells replaced:", n)


if __name__ == "__main__":
    main()
