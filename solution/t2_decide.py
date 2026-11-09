"""Window-level decision rule for Task 2: pick the predicted cell set that maximises the expected
window IoU under the model's probabilities, instead of one fixed threshold for every window.

Every scored window holds at least one queued cell (the selector requires a queue in the horizon;
onset windows have it exactly at T+30), so the truth samples are drawn conditional on a non-empty
queue and at least one cell is always predicted. Cells are treated as independent; the candidate
sets are the top-k cells by probability, k chosen by Monte-Carlo expected IoU.

  python t2_decide.py --set feb --tags v1 v9s v7
  python t2_decide.py --set march --tags v1 v9s v7     # emulated March windows: APRIL decisions only
"""
from __future__ import annotations

import argparse

import numpy as np

import t2_train as T


def eiou_mask(P: np.ndarray, n_samp: int = 400, floor: float = 0.01, rng=None, temp: float = 1.0) -> np.ndarray:
    """(6, L) probabilities -> boolean prediction maximising the Monte-Carlo expected IoU."""
    rng = rng if rng is not None else np.random.default_rng(0)
    p = P.ravel().astype(np.float64)
    if temp != 1.0:
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6)); lq = np.log(np.clip(1 - p, 1e-6, 1))
        p = 1 / (1 + np.exp(-(lp - lq) / temp)) * (p > 0)
    order = np.argsort(-p)
    ps = p[order]
    K = max(1, int((ps > floor).sum()))
    ps = ps[:K]
    Y = rng.random((n_samp, K)) < ps[None, :]
    keep = Y.any(axis=1)
    if keep.sum() < n_samp // 10:          # rare: the queue is almost surely outside the top cells
        Y = Y[:0]
    else:
        Y = Y[keep]
    out = np.zeros(p.shape, bool)
    if len(Y) == 0:
        out[order[0]] = True
        return out.reshape(P.shape)
    inter = np.cumsum(Y, axis=1)
    ytot = Y.sum(axis=1, keepdims=True)
    union = np.arange(1, K + 1)[None, :] + ytot - inter
    k = int(np.argmax((inter / union).mean(axis=0))) + 1
    out[order[:k]] = True
    return out.reshape(P.shape)


def decide(P, cond, rule="thr", thr_on=0.55, thr_onset=0.5, **kw):
    if rule == "eiou":
        if cond == "queue_onset":
            pred = np.zeros(P.shape, bool)
            pred[5] = eiou_mask(P[5:6], **kw)[0]
            return pred
        return eiou_mask(P, **kw)
    if cond == "queue_ongoing":
        pred = P >= thr_on
        if not pred.any():
            pred.ravel()[np.argmax(P)] = True
        return pred
    pred = np.zeros(P.shape, bool)
    s = P[5] >= thr_onset
    if not s.any():
        s[np.argmax(P[5])] = True
    pred[5] = s
    return pred


def score(probs, parts, **kw):
    res = {"queue_ongoing": [], "queue_onset": []}
    k = 0
    for z in parts:
        for y, m, cond in zip(z["y"], z["m"], z["cond"]):
            res[cond].append(T.iou(decide(probs[k], cond, **kw), y, m)); k += 1
    out = {c: round(float(np.mean(v)), 4) for c, v in res.items()}
    out["mean"] = round((out["queue_ongoing"] + out["queue_onset"]) / 2, 4)
    return out


if __name__ == "__main__":
    from t2_offeval import official_parts, window_probs
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="feb", choices=["feb", "march"])
    ap.add_argument("--tags", nargs="+", default=["v1", "v9s", "v7"])
    a = ap.parse_args()
    if a.set == "feb":
        parts = official_parts("2031-02")
    else:
        from t2_month_eval import load_parts
        parts = load_parts()
    allp = {t: window_probs(t, parts) for t in a.tags}
    if len(a.tags) > 1:
        allp["ens"] = [np.mean([allp[t][k] for t in a.tags], axis=0) for k in range(len(allp[a.tags[0]]))]
    for name, probs in allp.items():
        print(name, "thr .55/.5", score(probs, parts), "| thr .45/.4", score(probs, parts, thr_on=0.45, thr_onset=0.4),
              "| eiou", score(probs, parts, rule="eiou"), "| eiou T=1.5", score(probs, parts, rule="eiou", temp=1.5), flush=True)
