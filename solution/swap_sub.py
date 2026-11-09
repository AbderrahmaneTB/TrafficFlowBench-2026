"""Build a new upload file from an existing one, replacing only the chosen tasks.

Faster than make_sub.py when a probe changes one task (e.g. only Task 4)."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from common import RELEASE, WORK, PANELS, QUEUE_PANELS, SPLITS
from submission import key_index

SUBS = f"{WORK}/subs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="existing submission name in work/subs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--t1", help="state prediction folder in work/ (V, Q arrays)")
    ap.add_argument("--queue-from", help="take queue_pred from this submission")
    ap.add_argument("--t2tag", nargs="+", help="Task 2 model tag(s); several tags average their probabilities")
    ap.add_argument("--t2tag-onset", nargs="+", help="other tag(s) for the onset windows only (probe one condition)")
    ap.add_argument("--thr-on", type=float, default=0.55)
    ap.add_argument("--thr-onset", type=float, default=0.5)
    ap.add_argument("--onset-rel", type=float, help="onset links with P >= this share of the window's largest P "
                    "(replaces --thr-onset)")
    ap.add_argument("--t2full", action="store_true")
    ap.add_argument("--stable", choices=["hold"], help="ongoing windows the selector shows were stable through "
                    "T+25 (t2_stable.py): repeat the T+5 forecast for T+10..T+25")
    ap.add_argument("--queue-private-from", help="take queue_pred of the PRIVATE windows from this submission: April "
                    "windows may use settings chosen on March evidence, March windows only train-split evidence")
    ap.add_argument("--t4", help="Task 4 variant in work/t4")
    ap.add_argument("--zero-queue", choices=["all", "queue_onset", "queue_ongoing"],
                    help="leaderboard probe: blank the queue of these windows (their IoU becomes 0)")
    a = ap.parse_args()
    sub = pd.read_csv(f"{SUBS}/{a.base}.csv")
    meta = json.load(open(f"{SUBS}/{a.base}.json"))
    meta["base"] = a.base
    K = key_index()
    assert len(sub) == len(K["task"])
    if a.t1:
        sp = sub.speed_kmh.to_numpy().copy(); fl = sub.flow_vph.to_numpy().copy()
        for pi, p in enumerate(PANELS):
            for si, s in enumerate(SPLITS[1:], start=1):
                m = (K["task"] == 0) & (K["panel"] == pi) & (K["split"] == si)
                z = np.load(f"{WORK}/{a.t1}/{p}_{s}.npz")
                sp[m] = np.round(z["V"][K["d"][m], K["t"][m], K["l"][m]], 4)
                fl[m] = np.round(z["Q"][K["d"][m], K["t"][m], K["l"][m]], 3)
        st = K["task"] == 0
        assert np.isfinite(sp[st]).all() and (sp[st] > 0).all() and (fl[st] >= 0).all()
        assert float((fl[st] < 50).mean()) <= 0.2
        sub["speed_kmh"] = sp; sub["flow_vph"] = fl
        meta["t1"] = a.t1
    if a.queue_from:
        q = pd.read_csv(f"{SUBS}/{a.queue_from}.csv", usecols=["queue_pred"]).queue_pred.to_numpy()
        assert len(q) == len(sub)
        sub["queue_pred"] = q
        qm = json.load(open(f"{SUBS}/{a.queue_from}.json"))
        for k in ("t2tag", "thr_on", "thr_onset", "onset_mode", "on_mode", "k_min", "t2full"):
            meta[k] = qm.get(k)
    if a.t2tag:
        from t2_predict import predict as t2_predict
        if len(a.t2tag) == 1 and not a.t2tag_onset and a.onset_rel is None:
            queue, _ = t2_predict(a.t2tag[0], a.t2full, a.thr_on, a.thr_onset, "thr", 1)
        else:
            probs = {t: t2_predict(t, a.t2full, a.thr_on, a.thr_onset, "thr", 1)[1]
                     for t in set(a.t2tag) | set(a.t2tag_onset or [])}
            cond = pd.concat([pd.read_csv(f"{RELEASE}/task2/{p}/{s}/window_index.csv", usecols=["window_id", "condition"])
                              for p in QUEUE_PANELS for s in SPLITS[1:]]).set_index("window_id").condition
            queue = {}
            for w in probs[a.t2tag[0]]:
                tags = a.t2tag if cond[w] == "queue_ongoing" or not a.t2tag_onset else a.t2tag_onset
                P = np.mean([probs[t][w] for t in tags], axis=0)
                pred = np.zeros(P.shape, np.int8)
                if cond[w] == "queue_ongoing":
                    pred[:] = P >= a.thr_on
                else:
                    s = P[5] >= (a.onset_rel * P[5].max() if a.onset_rel is not None else a.thr_onset)
                    if not s.any():
                        s[np.argmax(P[5])] = True
                    pred[5] = s
                queue[w] = pred
        if a.stable == "hold":
            from t2_stable import stable_flags
            n = 0
            for p in QUEUE_PANELS:
                for s in SPLITS[1:]:
                    for w, fl in stable_flags(p, s).items():
                        if fl:
                            queue[w] = queue[w].copy(); queue[w][1:5] = queue[w][0]; n += 1
            meta["stable"] = f"hold ({n} windows)"
        mq = np.flatnonzero(K["task"] == 1)
        q = sub.queue_pred.to_numpy().copy()
        wids = K["wid"][mq]
        for w in np.unique(wids):
            r = mq[wids == w]
            q[r] = queue[w][K["h"][r] - 1, K["l"][r]]
        sub["queue_pred"] = q
        meta.update(t2tag=a.t2tag, t2tag_onset=a.t2tag_onset, thr_on=a.thr_on, thr_onset=a.thr_onset, t2full=a.t2full)
        if a.onset_rel is not None:
            meta["onset_rel"] = a.onset_rel
    if a.queue_private_from:
        q2 = pd.read_csv(f"{SUBS}/{a.queue_private_from}.csv", usecols=["queue_pred"]).queue_pred.to_numpy()
        assert len(q2) == len(sub)
        m = (K["task"] == 1) & (K["split"] == SPLITS.index("private"))
        q = sub.queue_pred.to_numpy().copy(); q[m] = q2[m]
        sub["queue_pred"] = q
        meta["queue_private_from"] = a.queue_private_from
    if a.zero_queue:
        cond = pd.concat([pd.read_csv(f"{RELEASE}/task2/{p}/{s}/window_index.csv", usecols=["window_id", "condition"])
                          for p in QUEUE_PANELS for s in SPLITS[1:]]).set_index("window_id").condition
        wc = pd.Series(K["wid"]).map(cond).to_numpy()
        m = (K["task"] == 1) & ((wc == a.zero_queue) if a.zero_queue != "all" else True)
        q = sub.queue_pred.to_numpy().copy(); q[m] = 0
        sub["queue_pred"] = q
        meta["zero_queue"] = a.zero_queue
    if a.t4:
        t4 = np.load(f"{WORK}/t4/{a.t4}.npz")
        pf = sub.path_flow.to_numpy().copy()
        for pi, p in enumerate(PANELS):
            for si, s in enumerate(SPLITS[1:], start=1):
                m = (K["task"] == 2) & (K["panel"] == pi) & (K["split"] == si)
                pf[m] = t4[f"{p}|{s}"][K["path"][m]]
        assert np.isfinite(pf).all() and (pf >= 0).all()
        sub["path_flow"] = pf
        meta["t4"] = a.t4
    out = f"{SUBS}/{a.name}.csv"
    sub.to_csv(out, index=False, float_format="%.6g")
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()[:12]
    meta.update(name=a.name, rows=len(sub), sha=sha, queue_positive=int(sub.queue_pred.sum()))
    json.dump(meta, open(f"{SUBS}/{a.name}.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
