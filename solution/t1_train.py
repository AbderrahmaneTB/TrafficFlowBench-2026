"""Train the Task 1 residual models on the train split's real masks.

Rows from June-January train the model; February is a held-out month scored with
the official per-regime formula. --full retrains on every month for submission.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np

from common import WORK, PANELS, load, network
from t1_features import build

FEAT = f"{WORK}/t1feat"
os.makedirs(FEAT, exist_ok=True)


def panel_rows(panel: str, n_train: int, n_hold: int, seed: int = 0):
    from t1_features import FEAT_VERSION
    path = f"{FEAT}/{panel}_{n_train}_{n_hold}{'' if FEAT_VERSION == 1 else f'_f{FEAT_VERSION}'}{f'_r{seed}' if seed else ''}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    z = load(panel, "train")
    dates = [str(x) for x in z["dates"]]
    from t1_features import target_index
    idx = target_index(panel, "train", dates, [str(x) for x in z["links"]])
    hold_days = np.array([s.startswith("2031-02") for s in dates])
    is_hold = hold_days[idx["d"]]
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(~is_hold); ho = np.flatnonzero(is_hold)
    tr = np.sort(rng.choice(tr, min(n_train, len(tr)), replace=False))
    ho = np.sort(rng.choice(ho, min(n_hold, len(ho)), replace=False))
    sel = np.concatenate([tr, ho])
    X, names, bv, bq, ix, (tv, tq) = build(panel, "train", z, sel)
    lanes = network(panel).lanes.to_numpy()
    out = dict(X=X, names=np.array(names), bv=bv, bq=bq, tv=tv.astype(np.float32), tq=tq.astype(np.float32),
               hold=np.r_[np.zeros(len(tr), bool), np.ones(len(ho), bool)], reg=ix["reg"],
               panel=np.full(len(sel), PANELS.index(panel), np.int8))
    np.savez(path, **out)
    return out


def state_score(pv, pq, tv, tq, reg):
    """Official S_state for one panel: per-regime RMSE scores, then their mean (flow per lane)."""
    s = []
    for r in range(3):
        m = reg == r
        if not m.any():
            continue
        rv = np.sqrt(np.mean((pv[m] - tv[m]) ** 2)); rq = np.sqrt(np.mean((pq[m] - tq[m]) ** 2))
        s.append(0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600))
    return float(np.mean(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=300_000)
    ap.add_argument("--n-hold", type=int, default=100_000)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--frac", type=float, default=1.0)
    a = ap.parse_args()
    parts = []
    for p in PANELS:
        t0 = time.time()
        parts.append(panel_rows(p, a.n_train, a.n_hold))
        print(p, "rows", len(parts[-1]["tv"]), f"{time.time() - t0:.0f}s", flush=True)
    names = list(parts[0]["names"])
    cat = {k: np.concatenate([q[k] for q in parts]) for k in ("X", "bv", "bq", "tv", "tq", "hold", "reg", "panel")}
    ok = np.isfinite(cat["tv"]) & np.isfinite(cat["tq"])
    if a.frac < 1:
        keep = np.random.default_rng(1).random(len(ok)) < a.frac
        ok &= keep
    tr = ok & (~cat["hold"] | a.full)
    ho = ok & cat["hold"]
    params = dict(objective="regression", learning_rate=a.lr, num_leaves=255, min_data_in_leaf=200,
                  feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=5.0,
                  max_bin=255, num_threads=a.threads, verbosity=-1, seed=7)
    preds = {}
    models = {}
    for tgt, base, truth in (("speed", "bv", "tv"), ("flow", "bq", "tq")):
        y = cat[truth] - cat[base]
        dtr = lgb.Dataset(cat["X"][tr], y[tr], feature_name=names, free_raw_data=False)
        dho = lgb.Dataset(cat["X"][ho], y[ho], reference=dtr)
        cb = [lgb.log_evaluation(250)]
        if not a.full:
            cb.append(lgb.early_stopping(100))
        m = lgb.train(params, dtr, num_boost_round=a.rounds, valid_sets=[dho], callbacks=cb)
        models[tgt] = m
        m.save_model(f"{WORK}/t1_{tgt}_{a.tag}{'_full' if a.full else ''}.txt")
        preds[tgt] = cat[base][ho] + m.predict(cat["X"][ho], num_threads=10)
        imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)[:15]
        print(tgt, "top features:", [n for _, n in imp], flush=True)
    # held-out scores per panel, official formula
    res = {}
    for pi, p in enumerate(PANELS):
        m = cat["panel"][ho] == pi
        tv, tq, rg = cat["tv"][ho][m], cat["tq"][ho][m], cat["reg"][ho][m]
        res[p] = dict(model=state_score(preds["speed"][m], preds["flow"][m], tv, tq, rg),
                      interp=state_score(cat["bv"][ho][m], cat["bq"][ho][m], tv, tq, rg),
                      rmse_v=float(np.sqrt(np.mean((preds["speed"][m] - tv) ** 2))),
                      rmse_q=float(np.sqrt(np.mean((preds["flow"][m] - tq) ** 2))))
        print(p, {k: round(v, 5) for k, v in res[p].items()}, flush=True)
    fam = {}
    for p in PANELS:
        fam.setdefault(p.rsplit("_", 1)[0], []).append(res[p]["model"])
    overall = float(np.mean([np.mean(v) for v in fam.values()]))
    print("held-out S_state (family mean):", round(overall, 6))
    json.dump(dict(res=res, overall=overall), open(f"{WORK}/t1_{a.tag}_holdout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
