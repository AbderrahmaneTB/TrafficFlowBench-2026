"""Fit the blank model on every emulated-blank row and predict the Task 1 targets inside the official blanks.

  TFB_T1FEAT=7 python t1_blank_final.py --out t1pred_blank [--rounds-v 1500 --rounds-q 1500]

Writes work/<out>/<panel>_<split>.npz (V, Q total flow) with values on the blank cells only; combine with a main
prediction folder through t1_blank_mix.py.
"""
from __future__ import annotations

import argparse
import os

import lightgbm as lgb
import numpy as np

from common import WORK, PANELS, QUEUE_PANELS, load, network
from t1_blank import real_rows, extra_features, extra2_features
from t1_blank_eval import PARAMS, collect, simple_flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="t1pred_blank")
    ap.add_argument("--rounds-v", type=int, default=1500)
    ap.add_argument("--rounds-q", type=int, default=1500)
    ap.add_argument("--groups", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ngroups", type=int, default=3)
    ap.add_argument("--train-ngroups", type=int, default=4)
    ap.add_argument("--train-groups", nargs="+", type=int, default=[0], help="train-month groups of emulated blanks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=["validation", "private"], help="splits whose emulated blanks train the model")
    ap.add_argument("--extra", action="store_true", help="second set of blank columns (t1_blank.extra_features)")
    ap.add_argument("--extra2", action="store_true", help="third set of blank columns (t1_blank.extra2_features)")
    ap.add_argument("--train-frac", type=float, default=1.0, help="share of the train-month days used (with --splits train ...)")
    ap.add_argument("--leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--min-leaf", type=int, default=PARAMS["min_data_in_leaf"])
    ap.add_argument("--flow-base", default="interp", choices=["interp", "simple"],
                    help="flow residual base: time interpolation, or the simple estimate (t1_blank_eval.simple_flow)")
    ap.add_argument("--flow-mix", type=float, default=1.0,
                    help="weight of the flow model; the rest goes to the simple flow estimate (CV: 0.5 best)")
    ap.add_argument("--simple", action="store_true",
                    help="no model: speed = time interpolation base, flow = mean of the base and the corridor-profile interpolation")
    a = ap.parse_args()
    models = {}
    if not a.simple:
        R, names = collect(a.splits, a.groups, a.ngroups, a.train_ngroups, extra=a.extra, train_frac=a.train_frac,
                           extra2=a.extra2, train_groups=a.train_groups)
        if a.flow_base == "simple":
            R["bq"] = simple_flow(R["X"], names, R["bq"])
        ok = np.isfinite(R["tv"]) & np.isfinite(R["tq"])
        X = R["X"] if ok.all() else R["X"][ok]          # no copy of the (large) matrix when every row is usable
        for tgt, base, truth, rounds in (("speed", "bv", "tv", a.rounds_v), ("flow", "bq", "tq", a.rounds_q)):
            ds = lgb.Dataset(X, (R[truth] - R[base])[ok])
            models[tgt] = lgb.train(dict(PARAMS, seed=a.seed, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf), ds, rounds)
            models[tgt].free_dataset()
            models[tgt].save_model(f"{WORK}/t1blank_{tgt}_{a.out}.lgb")
            print(tgt, "trained", rounds, "rounds on", int(ok.sum()), "rows", flush=True)
        del R, X
    os.makedirs(f"{WORK}/{a.out}", exist_ok=True)
    for p in PANELS:
        for s in ("validation", "private"):
            z = load(p, s)
            D, T, L = z["mv"].shape
            V = np.full((D, T, L), np.nan, np.float32); Q = np.full((D, T, L), np.nan, np.float32)
            if p in QUEUE_PANELS:
                r = real_rows(p, s)
                if a.extra and len(r["d"]):
                    z = load(p, s)
                    r["X"] = np.column_stack([r["X"], extra_features(z, z, {k: r[k] for k in ("d", "t", "l")}, p)])
                if a.extra2 and len(r["d"]):
                    z = load(p, s)
                    org = r["t"].astype(np.int64) - r["X"][:, list(r["names"]).index("b_pos")].astype(np.int64)
                    r["X"] = np.column_stack([r["X"], extra2_features(z, z, {k: r[k] for k in ("d", "t", "l")}, org, p)])
                if len(r["d"]):
                    lanes = network(p).lanes.to_numpy(np.float32)
                    d, t, l = (r[k].astype(np.int64) for k in ("d", "t", "l"))
                    qc = r["X"][:, list(r["names"]).index("b_q_corr")]
                    q_simple = np.where(np.isfinite(qc), (qc + r["bq"]) / 2, r["bq"])
                    if a.simple:
                        pv, pq = r["bv"], q_simple
                    else:
                        pv = r["bv"] + models["speed"].predict(r["X"], num_threads=8)
                        pq = (q_simple if a.flow_base == "simple" else r["bq"]) + models["flow"].predict(r["X"], num_threads=8)
                        pq = a.flow_mix * pq + (1 - a.flow_mix) * q_simple
                    V[d, t, l] = np.clip(pv, 3.0, 140.0)
                    Q[d, t, l] = np.clip(pq, 0.0, 3500.0) * lanes[l]
                    print(p, s, "blank targets", len(r["d"]), flush=True)
            np.savez_compressed(f"{WORK}/{a.out}/{p}_{s}.npz", V=V, Q=Q)


if __name__ == "__main__":
    main()
