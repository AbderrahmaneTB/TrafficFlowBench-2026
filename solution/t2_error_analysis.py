"""Error analysis of the final Task 2 recipe (v14 x 3 seeds, ongoing P >= 0.5, onset P >= 0.5 x window max) for the
report, on February official-like windows (train scenario, models fitted without February, leave-one-day-out priors)
and on emulated March windows (new scenario; evidence for April decisions only).

Onset: is the most likely link inside the true queue ("right bottleneck"), does the predicted set touch the true
queue at all ("hit"), and the IoU in each case.
Ongoing: IoU by what the true queue does over the horizon (dissolves / shrinks / grows / about stable at T+30), and
for the windows the selection procedure marks as stable through T+25.

  python t2_error_analysis.py   -> WORK/t2_error_analysis.json (+ printed table)
"""
from __future__ import annotations

import json

import numpy as np

import t2_train as T
from common import WORK
from t2_month_eval import load_parts
from t2_offeval import official_parts, window_probs

TAGS = ["v14", "v14s1", "v14s2"]


def analyse(parts):
    probs = [np.mean(ps, axis=0) for ps in zip(*[window_probs(t, parts) for t in TAGS])]
    onset, ongoing = [], []
    k = 0
    for z in parts:
        si = z["names"].index("sel_stable1")
        for X3, y, m, cond in zip(z["X"], z["y"], z["m"], z["cond"]):
            P = probs[k]; k += 1
            if cond == "queue_onset":
                pred = np.zeros(P.shape, bool)
                s = P[5] >= 0.5 * P[5].max()
                pred[5] = s
                t5 = (y[5] > 0) & m[5]
                onset.append(dict(iou=T.iou(pred, y, m), right=bool(t5[int(np.argmax(P[5]))]), n_pred=int(s.sum()),
                                  n_true=int(t5.sum()), hit=bool((s & t5).any())))
            else:
                pred = P >= 0.5
                t = (y > 0) & m
                sz = t.sum(1)
                kind = ("dissolves" if sz[-1] == 0 else "grows" if sz[-1] > 1.3 * max(sz[0], 1)
                        else "shrinks" if sz[-1] < 0.7 * sz[0] else "about stable")
                ongoing.append(dict(iou=T.iou(pred, y, m), kind=kind, sel_stable=bool(X3[0, si] > 0.5)))
    out = {"onset_n": len(onset), "onset_iou": float(np.mean([r["iou"] for r in onset])),
           "onset_right_share": float(np.mean([r["right"] for r in onset])),
           "onset_iou_right": float(np.mean([r["iou"] for r in onset if r["right"]] or [np.nan])),
           "onset_iou_wrong": float(np.mean([r["iou"] for r in onset if not r["right"]] or [np.nan])),
           "onset_hit_share": float(np.mean([r["hit"] for r in onset])),
           "onset_iou_hit": float(np.mean([r["iou"] for r in onset if r["hit"]] or [np.nan])),
           "onset_links_pred": float(np.mean([r["n_pred"] for r in onset])),
           "onset_links_true": float(np.mean([r["n_true"] for r in onset])),
           "ongoing_n": len(ongoing), "ongoing_iou": float(np.mean([r["iou"] for r in ongoing]))}
    for kind in ("about stable", "shrinks", "grows", "dissolves"):
        sub = [r["iou"] for r in ongoing if r["kind"] == kind]
        out[f"ongoing_{kind}"] = (len(sub), float(np.mean(sub)) if sub else float("nan"))
    for flag in (True, False):
        sub = [r["iou"] for r in ongoing if r["sel_stable"] == flag]
        out[f"ongoing_selector_{'stable' if flag else 'other'}"] = (len(sub), float(np.mean(sub)) if sub else float("nan"))
    return out


if __name__ == "__main__":
    res = {"february": analyse(official_parts("2031-02", data="11L")), "emulated_march": analyse(load_parts())}
    json.dump(res, open(f"{WORK}/t2_error_analysis.json", "w"), indent=1)
    for k, v in res.items():
        print(k)
        for a, b in v.items():
            print(f"  {a:28s} {b}")
