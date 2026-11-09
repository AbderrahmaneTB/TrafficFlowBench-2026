"""Weighted average of Task 1 prediction folders (work/<dir>/<panel>_<split>.npz with V, Q).

  python t1_avg_preds.py --out t1pred_cb34 t1pred_cb3:0.5 t1pred_cb4:0.5
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common import WORK, PANELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+", help="dir:weight")
    a = ap.parse_args()
    specs = [(s.split(":")[0], float(s.split(":")[1])) for s in a.inputs]
    tot = sum(w for _, w in specs)
    os.makedirs(f"{WORK}/{a.out}", exist_ok=True)
    for p in PANELS:
        for s in ("validation", "private"):
            zs = [(np.load(f"{WORK}/{d}/{p}_{s}.npz"), w / tot) for d, w in specs]
            fin = np.isfinite(zs[0][0]["V"])
            for z, _ in zs[1:]:
                assert (np.isfinite(z["V"]) == fin).all(), (p, s)
            V = sum(z["V"] * w for z, w in zs); Q = sum(z["Q"] * w for z, w in zs)
            np.savez_compressed(f"{WORK}/{a.out}/{p}_{s}.npz", V=V.astype(np.float32), Q=Q.astype(np.float32))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
