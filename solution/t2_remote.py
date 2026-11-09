"""Remote-queue additions for ongoing windows (Nov 6, after an independent audit of the code).

The ongoing model only scores its candidate links (t2_train.candidates: a queued link within CAND = 30 positions, or
a queue in the history); every other link is forecast free, so a new queue at a second bottleneck of an ongoing
window is never predicted (on held-out February windows 44 of 303 ongoing windows have one; adding them perfectly
would be worth +0.004 S_total). This module scores those remote links of the official ongoing windows:

  kind "head"  : remote-queue LightGBM heads (work/t2/remote_<tag>[_full].txt, nov5_tools/remote_head.py), fitted
                 only on remote links of ongoing TRAINING windows, per horizon step, same inputs as v14
  kind "onset" : the onset models of the given tags (the question "where does a new queue appear" is theirs);
                 the same probability is used for every step

Inputs are the window's own features (rows at or before its origin), built exactly as t2_predict builds them.
A spec opts in per split with  "remote": {"kind": "head" | "onset", "tags": [...], "tau": p, "steps": [h, ...]}
(steps 0-5 = T+5 ... T+30); make_final.queue_from_spec then adds remote links with probability >= tau at those
steps to the ongoing forecast. Without the key nothing changes."""
from __future__ import annotations

import json
import os

import numpy as np
import lightgbm as lgb

from common import QUEUE_PANELS, WORK
from t2_train import add_corridor, augment, blank, candidates, load_model

OUT = f"{WORK}/t2"
INPUT_TAG = "v14"     # the remote heads were fitted on v14's input layout


def _inputs(cfg: dict, split: str, panel: str):
    """Official windows of one split / panel -> list of (window_id, condition, X, names), as in t2_predict."""
    z = np.load(f"{OUT}/{split}{cfg.get('data', '')}_{panel}.npz", allow_pickle=True)
    names0 = list(z["names"])
    Xall = z["X"]
    if cfg.get("corr"):
        Xall, names0 = add_corridor(Xall, names0, int(cfg["corr"]))
    Xall = blank(Xall, names0, cfg.get("drop", []))
    out = []
    for X, wid, cond in zip(Xall, z["wid"], z["cond"]):
        names = names0
        if cfg.get("aug"):
            X, names = augment(X, names0)
        out.append((str(wid), str(cond), X, names))
    return out


def remote_probs(kind: str, tags: list[str], full: bool, splits=("validation", "private")) -> dict:
    """window_id -> (6, L) probabilities on the remote links of ongoing windows (0 on candidate links)."""
    sfx = "_full" if full else ""
    if kind == "head":
        cfgs = {t: json.load(open(f"{OUT}/config_{INPUT_TAG}.json")) for t in tags}
        models = {t: lgb.Booster(model_file=f"{OUT}/remote_{t}{sfx}.txt") for t in tags}
    elif kind == "onset":
        cfgs = {t: json.load(open(f"{OUT}/config_{t}.json")) for t in tags}
        models = {t: load_model("onset", cfgs[t].get("model_tag", t), full) for t in tags}
    else:
        raise ValueError(kind)
    acc: dict = {}
    for t in tags:
        for p in QUEUE_PANELS:
            for s in splits:
                for wid, cond, X, names in _inputs(cfgs[t], s, p):
                    if cond != "queue_ongoing":
                        continue
                    rem = ~candidates(X, names)
                    P = np.zeros((6, X.shape[0]), np.float32)
                    if rem.any():
                        Xr = X[rem]
                        if kind == "head":
                            for h in range(6):
                                P[h, rem] = models[t].predict(np.column_stack([Xr, np.full(len(Xr), h + 1, np.float32)]))
                        else:
                            P[:, rem] = models[t].predict(Xr)[None, :]
                    acc.setdefault(wid, []).append(P)
    return {w: np.mean(v, axis=0) for w, v in acc.items()}


def queued_at_origin(splits=("validation", "private")) -> dict:
    """window_id -> (L,) bool: candidate links that are queued at the origin itself (distance 0 to a queued link),
    for the optional per-band ongoing threshold "thr_on_q" of a spec (make_final.queue_from_spec)."""
    cfg = json.load(open(f"{OUT}/config_{INPUT_TAG}.json"))
    out = {}
    for p in QUEUE_PANELS:
        for s in splits:
            for wid, cond, X, names in _inputs(cfg, s, p):
                if cond != "queue_ongoing":
                    continue
                c = {n: i for i, n in enumerate(names)}
                dq = np.minimum(X[:, c["dist_q_down"]], X[:, c["dist_q_up"]])
                out[wid] = candidates(X, names) & (dq <= 0)
    return out


def model_files(remote: dict, full: bool = True) -> list[str]:
    """Files a remote spec needs (for run_task2's prerequisite check)."""
    sfx = "_full" if full else ""
    if remote["kind"] == "head":
        return [f"{OUT}/remote_{t}{sfx}.txt" for t in remote["tags"]] + [f"{OUT}/config_{INPUT_TAG}.json"]
    return [f"{OUT}/onset_{t}{sfx}.txt" for t in remote["tags"]]
