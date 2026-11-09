"""Predict Task 1 speed and flow at every template target of validation and private."""
from __future__ import annotations

import argparse
import os

import lightgbm as lgb
import numpy as np

from common import WORK, PANELS, load, network
from t1_features import build

OUT = f"{WORK}/t1pred"
os.makedirs(OUT, exist_ok=True)


def predict_panel(panel, split, models, chunk=400_000):
    z = load(panel, split)
    D, T, L = z["mv"].shape
    lanes = network(panel).lanes.to_numpy(np.float32)
    from t1_features import target_index
    idx = target_index(panel, split, [str(x) for x in z["dates"]], [str(x) for x in z["links"]])
    n = len(idx["d"])
    V = np.full((D, T, L), np.nan, np.float32); Q = np.full((D, T, L), np.nan, np.float32)
    for s in range(0, n, chunk):
        sel = np.arange(s, min(n, s + chunk))
        X, names, bv, bq, ix, _ = build(panel, split, z, sel)
        pv = bv + np.mean([m.predict(X, num_threads=10) for m in models["speed"]], axis=0)
        pq = bq + np.mean([m.predict(X, num_threads=10) for m in models["flow"]], axis=0)
        d, t, l = ix["d"], ix["t"], ix["l"]
        V[d, t, l] = np.clip(pv, 3.0, 140.0)
        Q[d, t, l] = np.clip(pq, 0.0, 3500.0) * lanes[l]
    np.savez(f"{OUT}/{panel}_{split}.npz", V=V, Q=Q)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v1"])
    a = ap.parse_args()
    models = {t: [lgb.Booster(model_file=f"{WORK}/t1_{t}_{m}.txt") for m in a.models] for t in ("speed", "flow")}
    for p in PANELS:
        for s in ("validation", "private"):
            print(p, s, predict_panel(p, s, models), flush=True)
