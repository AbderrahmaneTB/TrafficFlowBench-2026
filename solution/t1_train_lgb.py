"""Task 1 residual models with LightGBM (CPU): same rows, features, pseudo-target rows and February holdout as
t1_train_cb.py, a leaf-wise learner, so that its errors average out against the CatBoost / XGBoost models.

  python t1_train_lgb.py --tag lg1 --n-train 1200000 --n-hold 150000 --pseudo validation private
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np

from common import WORK, PANELS
from t1_train import panel_rows, state_score
from t1_train_cb import pseudo_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1_200_000)
    ap.add_argument("--n-hold", type=int, default=150_000)
    ap.add_argument("--rounds", type=int, default=6000)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--row-seed", type=int, default=0, help="sample of train-split target rows (0 = the cached one)")
    ap.add_argument("--tag", default="lg1")
    ap.add_argument("--pseudo", nargs="*", default=[])
    ap.add_argument("--pseudo-nmax", type=int, default=400_000)
    ap.add_argument("--pseudo-weight", type=float, default=2.0)
    ap.add_argument("--pseudo-hold", type=int, default=0)
    ap.add_argument("--pseudo-seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    parts = []
    for p in PANELS:
        t0 = time.time()
        parts.append(panel_rows(p, a.n_train, a.n_hold, a.row_seed))
        print(p, len(parts[-1]["tv"]), f"{time.time() - t0:.0f}s", flush=True)
    names = [str(x) for x in parts[0]["names"]]
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
    params = dict(objective="regression", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=5.0, max_bin=255,
                  num_threads=os.cpu_count(), verbosity=-1, seed=a.seed)
    preds = {}
    for tgt, base, truth in (("speed", "bv", "tv"), ("flow", "bq", "tq")):
        y = np.r_[(cat[truth] - cat[base])[tr], (ps[truth] - ps[base])[ptr]].astype(np.float32)
        dtr = lgb.Dataset(Xtr, y, weight=w, feature_name=names, free_raw_data=False)
        dho = lgb.Dataset(Xho, (cat[truth] - cat[base])[ho].astype(np.float32), reference=dtr)
        t0 = time.time()
        m = lgb.train(params, dtr, a.rounds, valid_sets=[dho],
                      callbacks=[lgb.log_evaluation(500), lgb.early_stopping(200)])
        print(tgt, "best iteration", m.best_iteration, f"{time.time() - t0:.0f}s", flush=True)
        m.save_model(f"{WORK}/t1_{tgt}_{a.tag}.lgb", num_iteration=m.best_iteration)
        preds[tgt] = cat[base][ho] + m.predict(Xho, num_iteration=m.best_iteration)
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
