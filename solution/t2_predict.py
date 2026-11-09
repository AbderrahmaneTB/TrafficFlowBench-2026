"""Task 2 inference on the official validation/private windows."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from common import WORK, QUEUE_PANELS
import json, os
from t2_train import candidates, expected_iou_set, augment, blank, add_corridor, load_model, RECENT_NAMES

OUT = f"{WORK}/t2"


def predict(tag="v1", full=True, thr_on=0.45, thr_onset=0.5, onset_mode="thr", k_min=1, splits=("validation", "private"),
            on_mode="thr"):
    cfg_path = f"{OUT}/config_{tag}.json"
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"aug": False, "recent": False}
    if cfg.get("nn"):
        from t2_nn import predict as nn_predict
        probs = nn_predict(tag, full, splits)
        cond = {}
        for p in QUEUE_PANELS:
            for s in splits:
                z = np.load(f"{OUT}/{s}{cfg.get('data', '')}_{p}.npz", allow_pickle=True)
                cond.update(zip(map(str, z["wid"]), z["cond"]))
        queue = {}
        for w, P in probs.items():
            pred = np.zeros(P.shape, np.int8)
            if cond[w] == "queue_ongoing":
                pred[:] = P >= thr_on
            else:
                sel = P[5] >= thr_onset
                if not sel.any():
                    sel[np.argmax(P[5])] = True
                pred[5] = sel
            queue[w] = pred
        return queue, probs
    # "model_tag": reuse another tag's model files on different inputs (e.g. v14F = v14 on data 11F, the month
    # features from the blank-filled calendar, t2_blankfill.py)
    m_on = load_model("ongoing", cfg.get("model_tag", tag), full)
    m_onset = load_model("onset", cfg.get("model_tag", tag), full)
    queue, probs = {}, {}
    for p in QUEUE_PANELS:
        for s in splits:
            z = np.load(f"{OUT}/{s}{cfg.get('data', '')}_{p}.npz", allow_pickle=True)
            names0 = list(z["names"])
            Xall = z["X"]
            if cfg["recent"]:
                R = np.load(f"{OUT}/recent_{s}_{p}.npz")["R"].astype(np.float32)
                Xall = np.concatenate([Xall, R], axis=2); names0 = names0 + RECENT_NAMES
            if cfg.get("corr"):
                Xall, names0 = add_corridor(Xall, names0, int(cfg["corr"]))
            if cfg.get("kin"):
                from t2_train import add_kin
                Xall, names0 = add_kin(Xall, names0)
            use_aug = cfg["aug"]
            Xall = blank(Xall, names0, cfg.get("drop", []))
            for X, wid, cond in zip(Xall, z["wid"], z["cond"]):
                names = names0
                if use_aug:
                    X, names = augment(X, names0)
                assert m_onset.num_feature() == len(names) and m_on.num_feature() == len(names) + 1
                L = X.shape[0]
                pred = np.zeros((6, L), np.int8)
                P = np.zeros((6, L), np.float32)
                if cond == "queue_ongoing":
                    cand = candidates(X, names)
                    if cand.any():
                        Xc = X[cand]
                        for h in range(6):
                            P[h, cand] = m_on.predict(np.column_stack([Xc, np.full(len(Xc), h + 1, np.float32)]))
                        if on_mode == "thr":
                            pred[:, cand] = P[:, cand] >= thr_on
                        else:
                            pred[:, cand] = expected_iou_set(P[:, cand].ravel()).reshape(6, -1)
                else:
                    pp = m_onset.predict(X)
                    P[5] = pp
                    if onset_mode == "thr":
                        sel = pp >= thr_onset
                        if sel.sum() < k_min:
                            sel = np.zeros(L, bool); sel[np.argsort(-pp)[:k_min]] = True
                    else:
                        sel = expected_iou_set(pp)
                    pred[5] = sel
                queue[str(wid)] = pred
                probs[str(wid)] = P
    return queue, probs
