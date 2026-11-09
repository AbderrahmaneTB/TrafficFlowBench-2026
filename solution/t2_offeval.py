"""Score Task 2 models on official-like February windows: one selector run per panel from Feb 1,
5 onset + 5 ongoing, exactly how the organizers draw a scored month (public = 40 + 40 such windows).

The v3 holdout (union of runs started every day) reads 0.89 for v1 ongoing while the leaderboard
reads 0.77; this checks whether the gap comes from which windows a month-start run picks."""
from __future__ import annotations

import argparse

import numpy as np

import t2_train as T
from common import QUEUE_PANELS, load, network
from t2_build import flat
from t2_data import HIST, select_official
from t2_underlying import underlying_queue
from t2_xeval import score


def official_parts(month="2031-02", data="9"):
    parts, pers = [], {"queue_ongoing": [], "queue_onset": []}
    for p in QUEUE_PANELS:
        z = load(p, "train")
        dates = [str(x) for x in z["dates"]]
        d0 = next(k for k, d in enumerate(dates) if d.startswith(month))
        d1 = max(k for k, d in enumerate(dates) if d.startswith(month)) + 1
        v = flat(z["v"]); el = flat(z["elig"]).astype(bool)
        vcut = network(p).vcut.to_numpy(np.float32)
        U = underlying_queue(v, vcut)
        wins = select_official(v, el, vcut, U, start=d0 * 288 + HIST, stop=d1 * 288 - 6)
        t3 = np.load(f"{T.OUT}/train{data}_{p}.npz", allow_pickle=True)
        key = {(int(i), str(c)): k for k, (i, c) in enumerate(zip(t3["meta_i"], t3["meta_cond"]))}
        rows = [key[(i, c)] for i, c in wins if (i, c) in key]
        miss = [(i, c) for i, c in wins if (i, c) not in key]
        parts.append(dict(X=t3["X"][rows], y=t3["y"][rows], m=t3["m"][rows], cond=t3["meta_cond"][rows],
                          names=list(t3["names"]), panel=p))
        for i, c in wins:
            if c == "queue_ongoing":
                st = (v[i - 1] <= vcut) & el[i - 1]          # official persistence: observed state at T-5
                y = U[i + 1:i + 7]; m = el[i + 1:i + 7] & np.isfinite(v[i + 1:i + 7])
                pers[c].append(T.iou(np.broadcast_to(st, (6, len(st))), y, m))
        print(p, "windows", len(wins), "found in v3", len(rows), "missing", miss, flush=True)
    print("official persistence (T-5) ongoing IoU:", round(float(np.mean(pers["queue_ongoing"])), 4))
    return parts


def window_probs(tag, parts):
    """Per window: (6, L) probabilities of one model (ongoing on candidates, onset at T+30)."""
    import json, os
    import lightgbm as lgb
    from t2_xeval import model_inputs
    cp = f"{T.OUT}/config_{tag}.json"
    cfg = json.load(open(cp)) if os.path.exists(cp) else {"aug": False, "recent": False}
    if cfg.get("nn"):
        from t2_nn import NNModels
        nm = NNModels(tag)
        return [nm.window(*model_inputs(X3, z["names"], cfg), z["panel"], cond)
                for z in parts for X3, cond in zip(z["X"], z["cond"])]
    m_on = T.load_model("ongoing", tag)
    m_onset = T.load_model("onset", tag)
    out = []
    for z in parts:
        for X3, cond in zip(z["X"], z["cond"]):
            X, names = model_inputs(X3, z["names"], cfg)
            P = np.zeros((6, X.shape[0]), np.float32)
            if cond == "queue_ongoing":
                cand = T.candidates(X, names)
                if cand.any():
                    P[:, cand] = np.stack([m_on.predict(np.column_stack([X[cand], np.full(cand.sum(), h + 1, np.float32)]))
                                           for h in range(6)])
            else:
                P[5] = m_onset.predict(X)
            out.append(P)
    return out


def ens_score(probs, parts, thr_on, thr_onset):
    res = {"queue_ongoing": [], "queue_onset": []}
    k = 0
    for z in parts:
        for y, m, cond in zip(z["y"], z["m"], z["cond"]):
            P = probs[k]; k += 1
            pred = np.zeros(P.shape, bool)
            if cond == "queue_ongoing":
                pred = P >= thr_on
            else:
                s = P[5] >= thr_onset
                if not s.any():
                    s[np.argmax(P[5])] = True
                pred[5] = s
            res[cond].append(T.iou(pred, y, m))
    out = {c: float(np.mean(v)) for c, v in res.items()}
    out["mean"] = (out["queue_ongoing"] + out["queue_onset"]) / 2
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["v1", "v2"])
    ap.add_argument("--month", default="2031-02")
    ap.add_argument("--ens", action="store_true", help="also score the equal-weight average of the tags")
    a = ap.parse_args()
    parts = official_parts(a.month)
    allp = {}
    for tag in a.tags:
        allp[tag] = window_probs(tag, parts)
        for thr in (0.45, 0.55, 0.65):
            r = ens_score(allp[tag], parts, thr, 0.5)
            print(tag, f"thr_on={thr}", {k: round(v, 4) for k, v in r.items()}, flush=True)
    if a.ens and len(a.tags) > 1:
        avg = [np.mean([allp[t][k] for t in a.tags], axis=0) for k in range(len(allp[a.tags[0]]))]
        for thr_on in (0.45, 0.55):
            for thr_onset in (0.4, 0.5, 0.6):
                r = ens_score(avg, parts, thr_on, thr_onset)
                print("ens", "+".join(a.tags), f"thr_on={thr_on} thr_onset={thr_onset}", {k: round(v, 4) for k, v in r.items()}, flush=True)
