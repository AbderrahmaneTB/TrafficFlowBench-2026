"""Evaluate Task 2 decoding choices on held-out February windows with saved models."""
from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

import t2_train as T
from common import QUEUE_PANELS

OUT = T.OUT


def load_hold(cfg):
    parts = []
    for p in QUEUE_PANELS:
        z = np.load(f"{OUT}/train{cfg.get('data', '')}_{p}.npz", allow_pickle=True)
        hold = np.flatnonzero(z["meta_hold"])
        X = z["X"][hold]; names = list(z["names"])
        if cfg["recent"]:
            R = np.load(f"{OUT}/recent_train_{p}.npz")["R"][hold].astype(np.float32)
            X = np.concatenate([X, R], axis=2); names = names + T.RECENT_NAMES
        if cfg["aug"]:
            aug = [T.augment(x, names) for x in X]
            X = np.stack([a[0] for a in aug]); names = aug[0][1]
        parts.append(dict(X=X, y=z["y"][hold], m=z["m"][hold], meta_cond=z["meta_cond"][hold],
                          meta_hold=np.ones(len(hold), bool), names=np.array(names), panel=p))
    return parts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    cp = f"{OUT}/config_{a.tag}.json"
    cfg = json.load(open(cp)) if os.path.exists(cp) else {"aug": False, "recent": False}
    parts = load_hold(cfg)
    names = list(parts[0]["names"])
    m_on = lgb.Booster(model_file=f"{OUT}/ongoing_{a.tag}.txt")
    m_onset = lgb.Booster(model_file=f"{OUT}/onset_{a.tag}.txt")
    cache = {}
    for thr in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
        r, b = T.evaluate(parts, names, m_on, m_onset, thr, thr, cache=cache)
        print(f"thr={thr}: model {r}  persistence {b}", flush=True)
    for k in (1, 2, 3):
        for thr in (0.3, 0.4, 0.5):
            r, _ = T.evaluate(parts, names, m_on, m_onset, 0.45, thr, k_min=k, cache=cache)
            print(f"onset thr={thr} k_min={k}: onset {r['queue_onset']:.4f}", flush=True)
    r, _ = T.evaluate(parts, names, m_on, m_onset, 0.45, 0.5, mode_onset="eiou", mode_on="eiou", cache=cache)
    print("expected-IoU decoding (both):", r, flush=True)
