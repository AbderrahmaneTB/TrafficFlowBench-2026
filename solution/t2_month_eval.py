"""Score Task 2 models on the emulated March windows (t2_month_windows.py) — for APRIL decisions only.

March data may inform April forecasts (it precedes every April origin) but never March's own windows.
March is a new scenario relative to the nine train months, so this is the closest available test of
how a model copes with April's unseen scenario.
"""
from __future__ import annotations

import argparse

import numpy as np

import t2_train as T
from common import QUEUE_PANELS
from t2_offeval import window_probs, ens_score
from t2_stable import SEL_NAMES


def load_parts(split="validation", max_day=None):
    """max_day keeps the windows of the first days of the month only (where the official windows sit)."""
    parts = []
    for p in QUEUE_PANELS:
        z = np.load(f"{T.OUT}/{split}M_{p}.npz", allow_pickle=True)
        k = np.ones(len(z["meta_cond"]), bool)
        if max_day is not None:
            k = np.array([int(str(d)[8:10]) <= max_day for d in z["meta_date"]])
        parts.append(dict(X=z["X"][k], y=z["y"][k], m=z["m"][k], cond=z["meta_cond"][k], names=list(z["names"]), panel=p))
    return parts


def hold_stable(P, parts):
    """Repeat the T+5 forecast through T+25 on windows the selector shows were stable."""
    out = []
    k = 0
    for z in parts:
        si = z["names"].index("sel_stable1")
        for X3, cond in zip(z["X"], z["cond"]):
            p = P[k].copy(); k += 1
            if cond == "queue_ongoing" and X3[0, si] > 0.5:
                p[1:5] = p[0]
            out.append(p)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["v1", "v9s", "v7"])
    ap.add_argument("--max-day", type=int)
    a = ap.parse_args()
    parts = load_parts(max_day=a.max_day)
    print("emulated March windows:", sum(len(z["cond"]) for z in parts))
    allp = {}
    for tag in a.tags:
        allp[tag] = window_probs(tag, parts)
        for thr in (0.45, 0.55):
            r = ens_score(allp[tag], parts, thr, 0.5)
            rh = ens_score(hold_stable(allp[tag], parts), parts, thr, 0.5)
            print(f"{tag:9s} thr_on={thr}: {r}  | stable-hold ongoing {rh['queue_ongoing']:.4f}", flush=True)
    if len(a.tags) > 1:
        avg = [np.mean([allp[t][k] for t in a.tags], axis=0) for k in range(len(allp[a.tags[0]]))]
        for thr in (0.45, 0.55):
            r = ens_score(avg, parts, thr, 0.5)
            rh = ens_score(hold_stable(avg, parts), parts, thr, 0.5)
            print(f"ens {'+'.join(a.tags)} thr_on={thr}: {r}  | stable-hold ongoing {rh['queue_ongoing']:.4f}", flush=True)
