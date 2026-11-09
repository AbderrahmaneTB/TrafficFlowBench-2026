"""Score Task 2 models of any version on the v3 held-out windows (official selector, underlying labels)."""
from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

import t2_train as T
from common import QUEUE_PANELS

OUT = T.OUT


def model_inputs(X3, names3, cfg):
    """Rebuild the input layout a model was trained with from a v3 window (L, F3)."""
    from t2_month import MONTH_NAMES
    from t2_stable import SEL_NAMES
    from t2_ramp import RAMP_NAMES
    ci = {n: i for i, n in enumerate(names3)}
    v3 = [n for n in names3 if n not in MONTH_NAMES and n not in SEL_NAMES and n not in RAMP_NAMES]
    base = [n for n in v3 if n not in ("slack", "since_prev", "n_prev_onset", "rq7_now", "rq7_h6", "rq7_dayany")]
    if cfg.get("data") == "12":
        names = v3 + [n for n in names3 if n in MONTH_NAMES] + SEL_NAMES + RAMP_NAMES
    elif cfg.get("data") in ("9", "11", "9L", "11L"):
        names = v3 + [n for n in names3 if n in MONTH_NAMES] + SEL_NAMES
    elif cfg.get("data") == "6":
        names = v3 + [n for n in names3 if n in MONTH_NAMES]
    elif cfg.get("data") == "3":
        names = v3
    else:
        names = base + (T.RECENT_NAMES if cfg.get("recent") else [])
    X = X3[:, [ci[n] for n in names]]
    if cfg.get("corr"):
        X = np.column_stack([X, T.corridor(X3, list(names3), int(cfg["corr"]))])
        names = names + T.corr_names(int(cfg["corr"]))
    if cfg.get("kin"):
        X = np.column_stack([X, T.kinematic(X3, list(names3))])
        names = names + T.KIN_NAMES
    X = T.blank(X, names, cfg.get("drop", []))
    if cfg.get("aug"):
        X, names = T.augment(X, names)
    return X, names


def score(tag, parts, thr_on, thr_onset, k_min=1, mode_on="thr", mode_onset="thr"):
    cp = f"{OUT}/config_{tag}.json"
    cfg = json.load(open(cp)) if os.path.exists(cp) else {"aug": False, "recent": False}
    m_on = T.load_model("ongoing", tag)
    m_onset = T.load_model("onset", tag)
    res = {"queue_ongoing": [], "queue_onset": []}
    per_panel = {}
    for z in parts:
        for X3, y, m, cond in zip(z["X"], z["y"], z["m"], z["cond"]):
            X, names = model_inputs(X3, z["names"], cfg)
            L = X.shape[0]
            pred = np.zeros((6, L), bool)
            if cond == "queue_ongoing":
                cand = T.candidates(X, names)
                if cand.any():
                    P = np.stack([m_on.predict(np.column_stack([X[cand], np.full(cand.sum(), h + 1, np.float32)]))
                                  for h in range(6)])
                    pred[:, cand] = (P >= thr_on) if mode_on == "thr" else T.expected_iou_set(P.ravel()).reshape(P.shape)
            else:
                p = m_onset.predict(X)
                if mode_onset == "thr":
                    s = p >= thr_onset
                    if s.sum() < k_min:
                        s = np.zeros(L, bool); s[np.argsort(-p)[:k_min]] = True
                else:
                    s = T.expected_iou_set(p)
                pred[5] = s
            v = T.iou(pred, y, m)
            res[cond].append(v)
            per_panel.setdefault((z["panel"], cond), []).append(v)
    out = {k: float(np.mean(v)) for k, v in res.items()}
    out["mean"] = (out["queue_ongoing"] + out["queue_onset"]) / 2
    return out, {f"{p}|{c[6:]}": round(float(np.mean(v)), 3) for (p, c), v in per_panel.items()}


def load_v3_hold():
    parts = []
    for p in QUEUE_PANELS:
        z = np.load(f"{OUT}/train9_{p}.npz", allow_pickle=True)
        h = np.flatnonzero(z["meta_hold"])
        parts.append(dict(X=z["X"][h], y=z["y"][h], m=z["m"][h], cond=z["meta_cond"][h],
                          names=list(z["names"]), panel=p))
    return parts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["v1"])
    a = ap.parse_args()
    parts = load_v3_hold()
    print("held-out windows:", {z["panel"]: len(z["cond"]) for z in parts})
    for tag in a.tags:
        for thr in (0.4, 0.5, 0.6):
            r, pp = score(tag, parts, thr, 0.5)
            print(tag, f"thr_on={thr}", {k: round(v, 4) for k, v in r.items()}, flush=True)
        print(tag, "per panel", pp)
