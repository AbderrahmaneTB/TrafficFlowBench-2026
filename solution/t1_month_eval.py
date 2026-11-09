"""Task 1 models scored on self-supervised pseudo-targets of a scored month: a local proxy of the
leaderboard's S_state on the new scenario (February, the usual holdout, runs on the train scenario).

  TFB_T1FEAT=1 python t1_month_eval.py --build            # cache feature-v1 pseudo rows
  TFB_T1FEAT=2 python t1_month_eval.py --build            # same cells, feature v2
  python t1_month_eval.py --models cb1:1 cb2f:2 cbf8_full:1 cbf10_full:1
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np

from common import WORK, PANELS, FAMILY
from t1_train import state_score

RHO, NMAX = 0.15, 150_000


def rows_path(p, split, ver):
    return f"{WORK}/t1feat/{p}_{split}_pseudo_{int(RHO * 100)}_{NMAX}_f{ver}.npz"


def family_mean(res):
    fams = sorted(set(FAMILY.values()))
    return float(np.mean([np.mean([res[p] for p in res if FAMILY[p] == f]) for f in fams if any(FAMILY[p] == f for p in res)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--models", nargs="*", default=[], help="tag:featureversion, CatBoost files t1_{speed,flow}_{tag}.cbm")
    a = ap.parse_args()
    if a.build:
        from t1_pseudo import pseudo_rows
        for p in PANELS:
            r = pseudo_rows(p, a.split, RHO, NMAX)
            print(p, a.split, len(r["tv"]), flush=True)
        raise SystemExit
    from catboost import CatBoostRegressor
    preds, truth = {}, {}
    for spec in a.models:
        tag, ver = spec.split(":")
        m = {}
        for t in ("speed", "flow"):
            m[t] = CatBoostRegressor(); m[t].load_model(f"{WORK}/t1_{t}_{tag}.cbm")
        for p in PANELS:
            z = np.load(rows_path(p, a.split, ver), allow_pickle=True)
            ok = np.isfinite(z["tv"]) & np.isfinite(z["tq"])
            pv = np.clip(z["bv"] + m["speed"].predict(z["X"], thread_count=10), 3, 140)
            pq = np.clip(z["bq"] + m["flow"].predict(z["X"], thread_count=10), 0, 3500)
            preds.setdefault(tag, {})[p] = (pv, pq, ok)
            truth[p] = (z["tv"], z["tq"], z["reg"])
        print(tag, "predicted", flush=True)
    tags = list(preds)
    combos = [(t,) for t in tags] + list(itertools.combinations(tags, 2)) + ([tuple(tags)] if len(tags) > 2 else [])
    for c in combos:
        res = {}
        for p in PANELS:
            tv, tq, reg = truth[p]
            ok = preds[c[0]][p][2]
            pv = np.mean([preds[t][p][0] for t in c], axis=0); pq = np.mean([preds[t][p][1] for t in c], axis=0)
            res[p] = state_score(pv[ok], pq[ok], tv[ok], tq[ok], reg[ok])
        print("+".join(c), f"{a.split} pseudo-target S_state (family mean) = {family_mean(res):.5f}",
              {p: round(v, 4) for p, v in res.items()}, flush=True)
