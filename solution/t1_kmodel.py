"""Experiment: a direct density model for Task 3.

S_LWR charges |dN_sub - dN_true| on transitions that touch a Task 1 target, with N = (q/v) * L.
For an isolated target that is 2 |N_hat - N_obs|, so what matters is the L1 error of q/v itself,
which separate speed and flow models do not minimise. This trains a LightGBM model of the
per-lane density residual on the cached Task 1 rows (June-January) with an L1 objective."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np

from common import WORK, PANELS
from t1_train import panel_rows

if __name__ == "__main__":
    Xs, ys, Xh, yh = [], [], [], []
    for p in PANELS:
        z = panel_rows(p, 300_000, 100_000)
        ok = np.isfinite(z["tv"]) & np.isfinite(z["tq"]) & (z["bv"] > 1)
        kb = z["bq"] / np.maximum(z["bv"], 1.0)
        kt = z["tq"] / np.maximum(z["tv"], 1.0)
        X = np.column_stack([z["X"], kb]).astype(np.float32)
        for sel, A, B in ((ok & ~z["hold"], Xs, ys), (ok & z["hold"], Xh, yh)):
            A.append(X[sel]); B.append((kt - kb)[sel])
    Xs, ys, Xh, yh = np.vstack(Xs), np.concatenate(ys), np.vstack(Xh), np.concatenate(yh)
    names = list(np.load(f"{WORK}/t1feat/{PANELS[0]}_300000_100000.npz", allow_pickle=True)["names"]) + ["k_base"]
    params = dict(objective="l1", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.7,
                  bagging_fraction=0.5, bagging_freq=1, num_threads=10, verbosity=-1, seed=7)
    d1 = lgb.Dataset(Xs, ys, feature_name=names); d2 = lgb.Dataset(Xh, yh, reference=d1)
    m = lgb.train(params, d1, 3000, valid_sets=[d2], callbacks=[lgb.log_evaluation(250), lgb.early_stopping(100)])
    m.save_model(f"{WORK}/t1_k_v1.txt")
    base = np.abs(yh).mean(); mod = np.abs(yh - m.predict(Xh, num_threads=10)).mean()
    print(f"held-out L1 of per-lane density: base {base:.4f} -> model {mod:.4f}", flush=True)
