"""Task 1 residual models with CatBoost on GPU (Kaggle T4). Same features and holdout as t1_train.

--pseudo adds self-supervised rows of the scored months (t1_pseudo): the train months share one
scenario, so only rows from the new scenarios teach the model when the month-own profiles should
override the train-wide ones. --pseudo-hold k keeps the rows of days d % k == 0 out of training and
scores them, next to the models named by --ref (Task 1 is offline, so this is all legal).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from catboost import CatBoostRegressor

from common import WORK, PANELS
from t1_train import panel_rows, state_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1_200_000)
    ap.add_argument("--n-hold", type=int, default=150_000)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--device", default="0")
    ap.add_argument("--task-type", default="GPU")
    ap.add_argument("--tag", default="cb1")
    ap.add_argument("--seed", type=int, default=20261023, help="CatBoost random_seed")
    ap.add_argument("--row-seed", type=int, default=0, help="sample of train-split target rows (0 = the cached one)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pseudo", nargs="*", default=[], help="scored splits whose pseudo-target rows join training")
    ap.add_argument("--pseudo-nmax", type=int, default=400_000)
    ap.add_argument("--pseudo-weight", type=float, default=2.0)
    ap.add_argument("--pseudo-seeds", type=int, nargs="+", default=[0], help="one extra hidden share per seed")
    ap.add_argument("--pseudo-hold", type=int, default=0)
    ap.add_argument("--ref", nargs="*", default=[], help="CatBoost tags scored on the held-out pseudo rows")
    a = ap.parse_args()
    parts = []
    for p in PANELS:
        t0 = time.time()
        parts.append(panel_rows(p, a.n_train, a.n_hold, a.row_seed))
        print(p, len(parts[-1]["tv"]), f"{time.time() - t0:.0f}s", flush=True)
    names = list(parts[0]["names"])
    cat = {k: np.concatenate([q[k] for q in parts]) for k in ("X", "bv", "bq", "tv", "tq", "hold", "reg", "panel")}
    del parts
    ok = np.isfinite(cat["tv"]) & np.isfinite(cat["tq"])
    tr = ok & (~cat["hold"] | a.full)
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
    del cat["X"]
    print("training rows", n1, "+ pseudo", n2, flush=True)
    preds = {}
    for tgt, base, truth in (("speed", "bv", "tv"), ("flow", "bq", "tq")):
        y = np.r_[(cat[truth] - cat[base])[tr], (ps[truth] - ps[base])[ptr]]
        m = CatBoostRegressor(iterations=a.iters, depth=a.depth, learning_rate=a.lr, l2_leaf_reg=10,
                              loss_function="RMSE", task_type=a.task_type, devices=a.device, border_count=254,
                              bootstrap_type="Bernoulli", subsample=0.8, random_seed=a.seed,
                              od_type="Iter", od_wait=300, verbose=500, **({"gpu_ram_part": 0.9} if a.task_type == "GPU" else {}))
        m.fit(Xtr, y, sample_weight=w, eval_set=(Xho, (cat[truth] - cat[base])[ho]), use_best_model=not a.full)
        m.save_model(f"{WORK}/t1_{tgt}_{a.tag}{'_full' if a.full else ''}.cbm")
        preds[tgt] = cat[base][ho] + m.predict(Xho)
        print(tgt, "best iteration", m.get_best_iteration(), flush=True)
    del Xtr
    if a.pseudo_hold:
        score_pseudo_hold(a, ps)
    res = {}
    for pi, p in enumerate(PANELS):
        mm = cat["panel"][ho] == pi
        tv, tq, rg = cat["tv"][ho][mm], cat["tq"][ho][mm], cat["reg"][ho][mm]
        res[p] = dict(model=state_score(preds["speed"][mm], preds["flow"][mm], tv, tq, rg),
                      interp=state_score(cat["bv"][ho][mm], cat["bq"][ho][mm], tv, tq, rg))
        print(p, res[p], flush=True)
    fam = {}
    for p in PANELS:
        fam.setdefault(p.rsplit("_", 1)[0], []).append(res[p]["model"])
    overall = float(np.mean([np.mean(v) for v in fam.values()]))
    print("held-out S_state (family mean):", round(overall, 6), flush=True)
    json.dump(dict(res=res, overall=overall), open(f"{WORK}/t1_{a.tag}_holdout.json", "w"), indent=1)


def pseudo_parts(a):
    keys = ("X", "bv", "bq", "tv", "tq", "reg", "d")
    if not a.pseudo:
        return dict(X=np.zeros((0, 1), np.float32), tv=np.zeros(0), tq=np.zeros(0), bv=np.zeros(0), bq=np.zeros(0),
                    phold=np.zeros(0, bool))
    from t1_pseudo import pseudo_rows
    parts = []
    for si, s in enumerate(a.pseudo):
        for p in PANELS:
            for seed in a.pseudo_seeds:
                r = pseudo_rows(p, s, 0.15, a.pseudo_nmax, seed)
                r = {k: r[k] for k in keys}
                r["panel"] = np.full(len(r["tv"]), PANELS.index(p), np.int8)
                r["split"] = np.full(len(r["tv"]), si, np.int8)
                r["phold"] = (r["d"] % a.pseudo_hold == 0) if a.pseudo_hold else np.zeros(len(r["tv"]), bool)
                print(p, s, "seed", seed, "pseudo rows", len(r["tv"]), "held out", int(r["phold"].sum()), flush=True)
                parts.append(r)
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


def score_pseudo_hold(a, ps):
    """S_state (family mean) on the held-out pseudo rows, per scored split, for this model and the --ref models."""
    hv = ps["phold"] & np.isfinite(ps["tv"]) & np.isfinite(ps["tq"])
    X = ps["X"][hv]
    tags = [a.tag + ("_full" if a.full else "")] + list(a.ref)
    for tag in tags:
        pv = ps["bv"][hv].copy(); pq = ps["bq"][hv].copy()
        m = CatBoostRegressor(); m.load_model(f"{WORK}/t1_speed_{tag}.cbm"); pv += m.predict(X)
        m = CatBoostRegressor(); m.load_model(f"{WORK}/t1_flow_{tag}.cbm"); pq += m.predict(X)
        pv = np.clip(pv, 3, 140); pq = np.clip(pq, 0, 3500)
        for si, s in enumerate(a.pseudo):
            fam = {}
            for pi, p in enumerate(PANELS):
                mm = (ps["split"][hv] == si) & (ps["panel"][hv] == pi)
                if mm.any():
                    fam.setdefault(p.rsplit("_", 1)[0], []).append(
                        state_score(pv[mm], pq[mm], ps["tv"][hv][mm], ps["tq"][hv][mm], ps["reg"][hv][mm]))
            print(f"{tag} {s} held-out pseudo S_state (family mean) = {np.mean([np.mean(v) for v in fam.values()]):.5f}", flush=True)


if __name__ == "__main__":
    main()
