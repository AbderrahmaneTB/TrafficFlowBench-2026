"""Assemble a final upload file from scratch, with separate Task 2 settings for March (validation) and April
(private) windows.

Task 2 settings for a window may only rest on evidence available before its origin: March windows on train-split
evidence, April windows also on March evidence (organizer ruling, forum 748789 / 749601). The spec says which
models (probabilities averaged) and thresholds each split and condition uses; full models (Feb included) by default.

  python make_final.py --name f1 --t1 t1pred_cb3 --spec final_spec.json
  final_spec.json: {"validation": {"onset": ["v9s", "v11"], "ongoing": ["v9s", "v11"], "thr_onset": 0.5, "thr_on": 0.5},
                    "private":    {...}}
  An optional "onset_rel": a per split replaces thr_onset by a * (largest onset probability of the window).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from common import RELEASE, WORK, PANELS, QUEUE_PANELS
from submission import assemble
from t2_predict import predict as t2_predict

SUBS = f"{WORK}/subs"


def queue_from_spec(spec: dict, full: bool) -> dict:
    tags = sorted({t for s in spec.values() for c in ("onset", "ongoing") for t in s[c]})
    probs = {t: t2_predict(t, full, 0.5, 0.5, "thr", 1)[1] for t in tags}
    remote = {}                       # optional remote-queue additions (t2_remote.py), per split
    for split, s in spec.items():
        if "remote" in s:
            from t2_remote import remote_probs
            r = s["remote"]
            remote[split] = remote_probs(r["kind"], r["tags"], full, splits=(split,))
    qnow = {}                         # optional "thr_on_q": own threshold for links queued at the origin
    for split, s in spec.items():
        if "thr_on_q" in s:
            from t2_remote import queued_at_origin
            qnow.update(queued_at_origin(splits=(split,)))
    queue = {}
    for split, s in spec.items():
        for p in QUEUE_PANELS:
            wi = pd.read_csv(f"{RELEASE}/task2/{p}/{split}/window_index.csv", usecols=["window_id", "condition"])
            for w, cond in zip(wi.window_id.astype(str), wi.condition):
                if cond == "queue_ongoing":
                    P = np.mean([probs[t][w] for t in s["ongoing"]], axis=0)
                    # thr_on_q: one value, or six (one per step T+5 ... T+30)
                    thr = (np.where(qnow[w][None, :], np.asarray(s["thr_on_q"], np.float32).reshape(-1, 1), s["thr_on"])
                           if "thr_on_q" in s else s["thr_on"])
                    pred = (P >= thr).astype(np.int8)
                    if not pred.any():
                        pred.ravel()[np.argmax(P)] = 1
                    if split in remote:
                        r = s["remote"]
                        for h in r.get("steps", range(6)):
                            pred[h] |= (remote[split][w][h] >= r["tau"]).astype(np.int8)
                else:
                    P = np.mean([probs[t][w] for t in s["onset"]], axis=0)
                    pred = np.zeros(P.shape, np.int8)
                    # optional relative rule: every link within a factor of the window's most likely link
                    sel = P[5] >= (s["onset_rel"] * P[5].max() if "onset_rel" in s else s["thr_onset"])
                    if not sel.any():
                        sel[np.argmax(P[5])] = True
                    pred[5] = sel
                queue[w] = pred
    return queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--t1", required=True, help="Task 1 prediction folder in work/")
    ap.add_argument("--spec", required=True, help="JSON file with per-split Task 2 settings")
    ap.add_argument("--t4", default="nnls")
    ap.add_argument("--not-full", action="store_true", help="use the Feb-held-out Task 2 models instead")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    assert set(spec) == {"validation", "private"}, "spec needs both splits"
    state = {}
    for p in PANELS:
        for s in ("validation", "private"):
            z = np.load(f"{WORK}/{a.t1}/{p}_{s}.npz")
            state[(p, s)] = (z["V"], z["Q"])
    queue = queue_from_spec(spec, not a.not_full)
    t4 = np.load(f"{WORK}/t4/{a.t4}.npz")
    odme = {(p, s): t4[f"{p}|{s}"] for p in PANELS for s in ("validation", "private")}
    os.makedirs(SUBS, exist_ok=True)
    out = f"{SUBS}/{a.name}.csv"
    sub = assemble(state, queue, odme, out)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()[:12]
    meta = dict(name=a.name, t1=a.t1, t4=a.t4, full=not a.not_full, spec=spec, rows=len(sub), sha=sha,
                queue_positive=int(sub.queue_pred.sum()), state_rows=int((sub.task == "state").sum()))
    json.dump(meta, open(f"{SUBS}/{a.name}.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
