"""Task 1 inference with CatBoost residual models (GPU if available)."""
from __future__ import annotations

import argparse
import os

import numpy as np
from catboost import CatBoostRegressor

from common import WORK, PANELS, load, network
from t1_features import build, target_index

# rows per feature build; every build refits the month (low-rank fits), so one chunk per panel-month is cheapest
CHUNK = 2_000_000


class _XGB:
    """XGBoost booster (t1_train_xgb.py) behind the CatBoost predict call used below."""

    def __init__(self, path):
        import xgboost as xgb
        self.b = xgb.Booster()
        self.b.load_model(path)
        self.b.set_param({"device": os.environ.get("TFB_XGB_DEVICE", "cpu")})

    def predict(self, X, task_type=None):
        return self.b.inplace_predict(X)


class _LGB:
    """LightGBM booster (t1_train_lgb.py) behind the same predict call."""

    def __init__(self, path):
        import lightgbm as lgb
        self.b = lgb.Booster(model_file=path)

    def predict(self, X, task_type=None):
        return self.b.predict(X, num_threads=os.cpu_count())


def _load(t: str, tag: str):
    path = f"{WORK}/t1_{t}_{tag}.ubj"
    if os.path.exists(path):
        return _XGB(path)
    if os.path.exists(f"{WORK}/t1_{t}_{tag}.lgb"):
        return _LGB(f"{WORK}/t1_{t}_{tag}.lgb")
    m = CatBoostRegressor()
    m.load_model(f"{WORK}/t1_{t}_{tag}.cbm")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["cb1"])
    ap.add_argument("--out", default="t1pred_cb1")
    ap.add_argument("--task-type", default="CPU")
    a = ap.parse_args()
    out = f"{WORK}/{a.out}"
    os.makedirs(out, exist_ok=True)
    models = {}
    for t in ("speed", "flow"):
        models[t] = []
        for tag in a.tags:
            models[t].append(_load(t, tag))
    for p in PANELS:
        for s in ("validation", "private"):
            z = load(p, s)
            D, T, L = z["mv"].shape
            lanes = network(p).lanes.to_numpy(np.float32)
            idx = target_index(p, s, [str(x) for x in z["dates"]], [str(x) for x in z["links"]])
            n = len(idx["d"])
            V = np.full((D, T, L), np.nan, np.float32); Q = np.full((D, T, L), np.nan, np.float32)
            for st in range(0, n, CHUNK):
                sel = np.arange(st, min(n, st + CHUNK))
                X, names, bv, bq, ix, _ = build(p, s, z, sel)
                pv = bv + np.mean([m.predict(X, task_type=a.task_type) for m in models["speed"]], axis=0)
                pq = bq + np.mean([m.predict(X, task_type=a.task_type) for m in models["flow"]], axis=0)
                V[ix["d"], ix["t"], ix["l"]] = np.clip(pv, 3.0, 140.0)
                Q[ix["d"], ix["t"], ix["l"]] = np.clip(pq, 0.0, 3500.0) * lanes[ix["l"]]
            np.savez_compressed(f"{out}/{p}_{s}.npz", V=V, Q=Q)
            print(p, s, n, flush=True)


if __name__ == "__main__":
    main()
