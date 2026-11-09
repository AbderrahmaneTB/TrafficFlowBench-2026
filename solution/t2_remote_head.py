"""Remote-queue head for ongoing windows (Nov 6, after an independent code audit): a LightGBM fitted ONLY on the links outside the
ongoing candidate gate (t2_train.candidates: no queued link within CAND positions, no queue in the history), per
horizon step h.  Same robust recipe and inputs as v14 (data 11, corridor level 2, window-relative augmentation,
link_pos / panel blanked, location priors rolled along the corridor in half of the training windows).
Fit = ongoing training windows outside the February hold-out; early stopping on the hold-out's remote rows.
  python t2_remote_head.py <seed> [--full]   -> work/t2/remote_rh{seed}[_full].txt (+ rounds_rh{seed}.json)
Inputs are the window's features (rows <= T); labels only from the training split."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lightgbm as lgb
import t2_train as T

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
full = "--full" in sys.argv
T.AUG, T.CORR, T.DATA = True, 2, "11"
T.DROP = ["link_pos", "panel"]
T.PRIOR_SHIFT = 0.5


def remote_rows(parts, names, use_hold):
    Xs, ys, ws = [], [], []
    for z in parts:
        sel = (z["meta_cond"] == "queue_ongoing") & (z["meta_hold"] == use_hold)
        X, y, m = z["X"][sel], z["y"][sel], z["m"][sel]
        cand = T.candidates(X, names)
        for w in range(len(X)):
            r = ~cand[w]
            for h in range(6):
                mm = r & m[w, h]
                if not mm.any():
                    continue
                Xs.append(np.column_stack([X[w][mm], np.full(mm.sum(), h + 1, np.float32)]))
                ys.append(y[w, h][mm])
                ws.append(np.full(mm.sum(), 1.0 / max(m[w].sum(), 1), np.float32))
    return np.vstack(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(ws) * 1000


parts = T.load_all(T.MAX_ONGOING, seed=seed)
names = list(parts[0]["names"])
Xa, ya, wa = remote_rows(parts, names, False)
Xb, yb, wb = remote_rows(parts, names, True)
del parts
print("remote rows fit", Xa.shape, f"pos {ya.mean():.4f}", "hold", Xb.shape, f"pos {yb.mean():.4f}", flush=True)
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=10, verbosity=-1, seed=11 + seed)
tag = f"rh{seed}"
if full:
    rounds = json.load(open(f"{T.OUT}/rounds_{tag}.json"))["remote"]
    Xa, ya, wa = np.vstack([Xa, Xb]), np.concatenate([ya, yb]), np.concatenate([wa, wb])
    m = lgb.train(params, lgb.Dataset(Xa, ya, weight=wa, feature_name=names + ["h"]), num_boost_round=rounds)
    m.save_model(f"{T.OUT}/remote_{tag}_full.txt")
else:
    d1 = lgb.Dataset(Xa, ya, weight=wa, feature_name=names + ["h"])
    d2 = lgb.Dataset(Xb, yb, weight=wb, reference=d1)
    m = lgb.train(params, d1, num_boost_round=4000, valid_sets=[d2],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)])
    m.save_model(f"{T.OUT}/remote_{tag}.txt")
    json.dump({"remote": m.best_iteration}, open(f"{T.OUT}/rounds_{tag}.json", "w"))
    imp = sorted(zip(m.feature_importance("gain"), names + ["h"]), reverse=True)[:20]
    print("remote top:", [n for _, n in imp], flush=True)
print("saved", tag, "full" if full else "", flush=True)
