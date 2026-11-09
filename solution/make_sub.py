"""Assemble one complete upload file from the chosen per-task predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from common import WORK, PANELS
from submission import assemble
from t2_predict import predict as t2_predict

SUBS = f"{WORK}/subs"
os.makedirs(SUBS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--t1", default="t1pred")
    ap.add_argument("--t2tag", default="v1")
    ap.add_argument("--thr-on", type=float, default=0.45)
    ap.add_argument("--thr-onset", type=float, default=0.5)
    ap.add_argument("--onset-mode", default="thr")
    ap.add_argument("--on-mode", default="thr")
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--t2full", action="store_true")
    ap.add_argument("--t4", default="kl")
    a = ap.parse_args()
    state = {}
    for p in PANELS:
        for s in ("validation", "private"):
            z = np.load(f"{WORK}/{a.t1}/{p}_{s}.npz")
            state[(p, s)] = (z["V"], z["Q"])
    queue, _ = t2_predict(a.t2tag, a.t2full, a.thr_on, a.thr_onset, a.onset_mode, a.k_min, on_mode=a.on_mode)
    t4 = np.load(f"{WORK}/t4/{a.t4}.npz")
    odme = {(p, s): t4[f"{p}|{s}"] for p in PANELS for s in ("validation", "private")}
    out = f"{SUBS}/{a.name}.csv"
    sub = assemble(state, queue, odme, out)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()[:12]
    meta = dict(vars(a), rows=len(sub), sha=sha,
                queue_positive=int(sub.queue_pred.sum()), state_rows=int((sub.task == "state").sum()))
    json.dump(meta, open(f"{SUBS}/{a.name}.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
