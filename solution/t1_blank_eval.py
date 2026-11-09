"""Emulated-blank evaluation of Task 1 predictors (rows from t1_blank_build.py).

Scored-month rows are split by day parity: a blank model is fitted on the train-month rows + the scored rows
of the other parity and scored on the held-out parity, next to the base interpolation, the corridor-profile
interpolations and (with --ref) the main LightGBM models of a tag applied to the same rows. Cells that were
pseudo-targets of the main models (t1_pseudo seed 0, 15 %) are left out of every score.

  TFB_T1FEAT=7 python t1_blank_eval.py --ref lg2 [--groups 0 1 2] [--rounds 3000]
"""
from __future__ import annotations

import argparse
import zlib

import lightgbm as lgb
import numpy as np

from common import WORK, QUEUE_PANELS, load
from t1_blank import blank_rows, blank_extra, blank_extra2, EXTRA_NAMES, EXTRA2_NAMES, FEAT
from t1_features import FEAT_VERSION

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8)


def pseudo_mask(panel: str, split: str, rho: float = 0.15) -> np.ndarray:
    z = load(panel, split)
    rng = np.random.default_rng(0 + zlib.crc32(f"{panel}|{split}".encode()))
    obs = z["melig"].astype(bool) & np.isfinite(z["mv"]) & np.isfinite(z["mq"])
    return obs & (rng.random(obs.shape) < rho)


def collect(splits, groups, ngroups, train_ngroups, panels=QUEUE_PANELS, extra=False, train_frac=1.0, extra2=False,
            train_groups=(0,)):
    """Emulated-blank rows of the given splits / groups. Two passes (row counts, then filling one preallocated matrix),
    so the peak memory is the result plus one group instead of twice the result."""
    specs = []
    for s in splits:
        for p in panels:
            gs, ng = (tuple(train_groups), train_ngroups) if s == "train" else (groups, ngroups)
            specs += [(s, p, g, ng) for g in gs]

    def keep_rows(s, d):
        if s == "train" and train_frac < 1:                    # whole days, so a blank stays complete
            return (d.astype(np.int64) * 2654435761 % 1000) < train_frac * 1000
        return np.ones(len(d), bool)

    sizes = []
    for s, p, g, ng in specs:
        blank_rows(p, s, g, ng)                                 # builds the cache file if missing
        with np.load(f"{FEAT}/{p}_{s}_blank_g{g}of{ng}_f{FEAT_VERSION}.npz", allow_pickle=True) as z:
            sizes.append(int(keep_rows(s, z["d"]).sum()))
    names = (list(blank_rows(panels[0], "validation", groups[0], ngroups)["names"]) + (EXTRA_NAMES if extra else [])
             + (EXTRA2_NAMES if extra2 else []))
    N, F = sum(sizes), len(names)
    out = {"X": np.empty((N, F), np.float32)}
    for k, dt in (("bv", np.float32), ("bq", np.float32), ("tv", np.float32), ("tq", np.float32), ("d", np.int16),
                  ("split", np.int8), ("panel", np.int8), ("pseudo", bool)):
        out[k] = np.empty(N, dt)
    pos = 0
    pm_cache = {}
    for (s, p, g, ng), n in zip(specs, sizes):
        r = blank_rows(p, s, g, ng)
        d = r["d"].astype(np.int64)
        keep = keep_rows(s, d)
        X = r["X"][keep]
        if extra:
            X = np.column_stack([X, blank_extra(p, s, g, ng)[keep]])
        if extra2:
            X = np.column_stack([X, blank_extra2(p, s, g, ng)[keep]])
        sl = slice(pos, pos + n)
        out["X"][sl] = X
        for k in ("bv", "bq", "tv", "tq"):
            out[k][sl] = r[k][keep]
        out["d"][sl] = d[keep]
        out["split"][sl] = {"train": 0, "validation": 1, "private": 2}[s]
        out["panel"][sl] = QUEUE_PANELS.index(p)
        if s != "train":
            if (p, s) not in pm_cache:
                pm_cache = {(p, s): pseudo_mask(p, s)}
            t, l = r["t"].astype(np.int64), r["l"].astype(np.int64)
            out["pseudo"][sl] = pm_cache[(p, s)][d, t, l][keep]
        else:
            out["pseudo"][sl] = False
        pos += n
        del r, X
    return out, names


def simple_flow(X, names, bq):
    """Mean of the time interpolation and the corridor-profile interpolation (the time interpolation alone where the
    corridor estimate is missing)."""
    qc = X[:, list(names).index("b_q_corr")]
    return np.where(np.isfinite(qc), (qc + bq) / 2, bq).astype(np.float32)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--groups", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ngroups", type=int, default=3)
    ap.add_argument("--train-ngroups", type=int, default=4)
    ap.add_argument("--train-groups", nargs="+", type=int, default=[0], help="train-month groups of emulated blanks")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--no-train-months", action="store_true")
    ap.add_argument("--panels", nargs="+", default=QUEUE_PANELS)
    ap.add_argument("--extra", action="store_true", help="add the second set of blank columns (t1_blank.extra_features)")
    ap.add_argument("--extra2", action="store_true", help="add the third set (t1_blank.extra2_features: queue geometry)")
    ap.add_argument("--train-frac", type=float, default=1.0, help="share of the train-month days used")
    ap.add_argument("--leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--lr", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--min-leaf", type=int, default=PARAMS["min_data_in_leaf"])
    ap.add_argument("--flow-base", default="interp", choices=["interp", "simple"],
                    help="flow residual base: time interpolation, or its mean with the corridor-profile interpolation")
    a = ap.parse_args()
    splits = (["validation", "private"] if a.no_train_months else ["train", "validation", "private"])
    R, names = collect(splits, a.groups, a.ngroups, a.train_ngroups, a.panels, a.extra, a.train_frac, a.extra2,
                       a.train_groups)
    R["bq0"] = R["bq"].copy()                  # time interpolation, kept for the simple flow estimate
    if a.flow_base == "simple":
        R["bq"] = simple_flow(R["X"], names, R["bq"])
    X = R["X"]
    nb = X.shape[1]
    print("rows", {s: int((R["split"] == i).sum()) for i, s in enumerate(("train", "validation", "private"))},
          "features", nb, flush=True)
    ok = np.isfinite(R["tv"]) & np.isfinite(R["tq"])
    refs = {}
    for tag in a.ref:
        m_v = lgb.Booster(model_file=f"{WORK}/t1_speed_{tag}.lgb"); m_q = lgb.Booster(model_file=f"{WORK}/t1_flow_{tag}.lgb")
        Xm = X[:, :m_v.num_feature()]         # valid for v7 models only: the blank columns come after the v7 ones
        assert m_v.num_feature() == 101, "ref must be a feature-v7 model"
        refs[tag] = (np.clip(R["bv"] + m_v.predict(Xm, num_threads=8), 3, 140), np.clip(R["bq"] + m_q.predict(Xm, num_threads=8), 0, 3500))
    pred_v = np.full(len(ok), np.nan, np.float32); pred_q = np.full(len(ok), np.nan, np.float32)
    scored = R["split"] > 0
    # early stopping on whole days of the training side (a hash of split, panel, day), never on the test fold
    es_day = (R["split"].astype(np.int64) * 7919 + R["panel"].astype(np.int64) * 104729 + R["d"].astype(np.int64) * 31) % 5 == 0
    for fold in (0, 1):
        te = scored & (R["d"] % 2 == fold) & ok
        side = ok & ~(scored & (R["d"] % 2 == fold))
        tr, va = side & ~es_day, side & es_day
        for tgt, base, truth, out in (("speed", "bv", "tv", pred_v), ("flow", "bq", "tq", pred_q)):
            dtr = lgb.Dataset(X[tr], (R[truth] - R[base])[tr], free_raw_data=False)
            dva = lgb.Dataset(X[va], (R[truth] - R[base])[va], reference=dtr)
            params = dict(PARAMS, num_leaves=a.leaves, learning_rate=a.lr, min_data_in_leaf=a.min_leaf)
            m = lgb.train(params, dtr, a.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
            out[te] = R[base][te] + m.predict(X[te], num_iteration=m.best_iteration, num_threads=8)
            print(f"fold {fold} {tgt} best iteration {m.best_iteration}", flush=True)
            if fold == 0 and tgt == "speed":
                imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)[:15]
                print("top gain:", ", ".join(f"{n}" for _, n in imp), flush=True)
    pred_v = np.clip(pred_v, 3, 140); pred_q = np.clip(pred_q, 0, 3500)
    col = {n: i for i, n in enumerate(names)}
    for si, s in ((1, "validation"), (2, "private")):
        ev = (R["split"] == si) & ok & ~R["pseudo"]
        tv, tq = R["tv"][ev], R["tq"][ev]
        line = [f"{s:10s} n {int(ev.sum())}",
                f"base {rmse(R['bv'][ev], tv):.2f} / {rmse(R['bq'][ev], tq):.1f}",
                f"corr-profile {rmse(X[ev, col['b_v_corr']], tv):.2f} / {rmse(X[ev, col['b_q_corr']], tq):.1f}",
                f"BLANK MODEL {rmse(pred_v[ev], tv):.2f} / {rmse(pred_q[ev], tq):.1f}",
                f"flow 0.7 model + 0.3 simple {rmse(0.7 * pred_q[ev] + 0.3 * simple_flow(X[ev], names, R['bq0'][ev]), tq):.1f}"]
        for tag, (rv, rq) in refs.items():
            line.append(f"{tag} {rmse(rv[ev], tv):.2f} / {rmse(rq[ev], tq):.1f}")
            line.append(f"avg(blank,{tag}) {rmse((pred_v[ev] + rv[ev]) / 2, tv):.2f} / {rmse((pred_q[ev] + rq[ev]) / 2, tq):.1f}")
        print(" | ".join(line), flush=True)
        for c in (1.0, 0.0):
            m = ev & (X[:, col["b_cond"]] == c)
            if m.any():
                txt = f"   {'onset' if c else 'ongoing'} n {int(m.sum())}: model {rmse(pred_v[m], R['tv'][m]):.2f} / {rmse(pred_q[m], R['tq'][m]):.1f}"
                for tag, (rv, rq) in refs.items():
                    txt += f" | {tag} {rmse(rv[m], R['tv'][m]):.2f} / {rmse(rq[m], R['tq'][m]):.1f}"
                print(txt, flush=True)
    np.savez(f"{WORK}/t1_blank_eval_preds.npz", pred_v=pred_v, pred_q=pred_q, **{f"ref_{k}_v": v[0] for k, v in refs.items()},
             **{f"ref_{k}_q": v[1] for k, v in refs.items()})


if __name__ == "__main__":
    main()
