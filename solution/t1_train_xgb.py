"""Task 1 residual models with XGBoost on GPU: same rows, features, pseudo-target rows and February holdout as
t1_train_cb.py, a different learner (depth-wise trees), so that its errors average out against the CatBoost models.

  python t1_train_xgb.py --tag xg1 --n-train 1200000 --n-hold 150000 --pseudo validation private
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import xgboost as xgb

from common import WORK, PANELS
from t1_train import panel_rows, state_score
from t1_train_cb import pseudo_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1_200_000)
    ap.add_argument("--n-hold", type=int, default=150_000)
    ap.add_argument("--rounds", type=int, default=10000)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="xg1")
    ap.add_argument("--pseudo", nargs="*", default=[])
    ap.add_argument("--pseudo-nmax", type=int, default=400_000)
    ap.add_argument("--pseudo-weight", type=float, default=2.0)
    ap.add_argument("--pseudo-hold", type=int, default=0)
    ap.add_argument("--pseudo-seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    parts = []
    for p in PANELS:
        t0 = time.time()
        parts.append(panel_rows(p, a.n_train, a.n_hold))
        print(p, len(parts[-1]["tv"]), f"{time.time() - t0:.0f}s", flush=True)
    cat = {k: np.concatenate([q[k] for q in parts]) for k in ("X", "bv", "bq", "tv", "tq", "hold", "reg", "panel")}
    del parts
    ok = np.isfinite(cat["tv"]) & np.isfinite(cat["tq"])
    tr = ok & ~cat["hold"]
    ho = ok & cat["hold"]
    ps = pseudo_parts(a)
    ptr = np.isfinite(ps["tv"]) & np.isfinite(ps["tq"]) & ~ps["phold"]
    n1, n2 = int(tr.sum()), int(ptr.sum())
    Xtr = np.empty((n1 + n2, cat["X"].shape[1]), np.float32)
    np.compress(tr, cat["X"], axis=0, out=Xtr[:n1])
    if n2:
        np.compress(ptr, ps["X"], axis=0, out=Xtr[n1:])
    w = np.r_[np.ones(n1, np.float32), np.full(n2, a.pseudo_weight, np.float32)]
    Xho = cat["X"][ho]
    del cat["X"], ps["X"]
    print("training rows", n1, "+ pseudo", n2, flush=True)
    params = dict(tree_method="hist", device=a.device, max_depth=a.depth, learning_rate=a.lr, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=100, reg_lambda=10.0, max_bin=256,
                  objective="reg:squarederror", eval_metric="rmse", seed=a.seed)
    preds = {}
    for tgt, base, truth in (("speed", "bv", "tv"), ("flow", "bq", "tq")):
        y = np.r_[(cat[truth] - cat[base])[tr], (ps[truth] - ps[base])[ptr]].astype(np.float32)
        dtr = xgb.QuantileDMatrix(Xtr, y, weight=w, max_bin=256)
        dho = xgb.QuantileDMatrix(Xho, (cat[truth] - cat[base])[ho].astype(np.float32), ref=dtr)
        t0 = time.time()
        bst = xgb.train(params, dtr, num_boost_round=a.rounds, evals=[(dho, "hold")], early_stopping_rounds=300,
                        verbose_eval=500)
        print(tgt, "best iteration", bst.best_iteration, f"{time.time() - t0:.0f}s", flush=True)
        bst = bst[: bst.best_iteration + 1]
        bst.save_model(f"{WORK}/t1_{tgt}_{a.tag}.ubj")
        preds[tgt] = cat[base][ho] + bst.inplace_predict(Xho)
        del dtr, dho
    res = {}
    for pi, p in enumerate(PANELS):
        mm = cat["panel"][ho] == pi
        tv, tq, rg = cat["tv"][ho][mm], cat["tq"][ho][mm], cat["reg"][ho][mm]
        res[p] = dict(model=state_score(preds["speed"][mm], preds["flow"][mm], tv, tq, rg))
        print(p, res[p], flush=True)
    fam = {}
    for p in PANELS:
        fam.setdefault(p.rsplit("_", 1)[0], []).append(res[p]["model"])
    overall = float(np.mean([np.mean(v) for v in fam.values()]))
    print("held-out S_state (family mean):", round(overall, 6), flush=True)
    json.dump(dict(res=res, overall=overall), open(f"{WORK}/t1_{a.tag}_holdout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
