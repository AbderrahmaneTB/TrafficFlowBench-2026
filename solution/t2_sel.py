"""Task 2 v9 datasets: v6 windows plus the selector-implied features of t2_stable.selection_features.

Rows before T are taken as a scored window sees them: the released history (T-60..T-5) raw, older
rows from the masked layer. Training windows get the same treatment from the train split, so the
feature means the same thing in both places. Only window origins before T enter `slack`.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import RELEASE, WORK, QUEUE_PANELS, load, network
from t2_build import flat
from t2_data import HIST
from t2_stable import MAXBACK, SEL_NAMES, selection_features

OUT = f"{WORK}/t2"
NBACK = MAXBACK + HIST + 1


def rows_before(i, raw_v, raw_el, m_v, m_el, vcut):
    """(Q, F) for the NBACK rows ending at T-5: raw for the last HIST rows, masked before."""
    lo = max(i - NBACK, 0)
    v = m_v[lo:i].copy(); e = m_el[lo:i].copy()
    k = min(HIST, i - lo)
    v[-k:] = raw_v[-k:]; e[-k:] = raw_el[-k:]
    F = np.isfinite(v) & e
    return F & (v <= vcut), F


def build(panel: str):
    vcut = network(panel).vcut.to_numpy(np.float32)
    z = load(panel, "train")
    v, el = flat(z["v"]), flat(z["elig"]).astype(bool)
    mv, mel = flat(z["mv"]), flat(z["melig"]).astype(bool)
    t6 = dict(np.load(f"{OUT}/train6_{panel}.npz", allow_pickle=True))
    si = list(t6["names"]).index("slack")
    feats = []
    for w, (i, cond) in enumerate(zip(t6["meta_i"], t6["meta_cond"])):
        i = int(i)
        if cond != "queue_ongoing":
            feats.append(np.zeros(len(SEL_NAMES), np.float32)); continue
        Q, F = rows_before(i, v[i - HIST:i], el[i - HIST:i], mv, mel, vcut)
        feats.append(selection_features(Q, F, float(t6["X"][w][0, si])))
    S = np.stack(feats)
    L = t6["X"].shape[1]
    t6["X"] = np.concatenate([t6["X"], np.repeat(S[:, None, :], L, axis=1)], axis=2)
    t6["names"] = np.array(list(t6["names"]) + SEL_NAMES)
    np.savez(f"{OUT}/train9_{panel}.npz", **t6)
    print(panel, "train", "stable1 share among ongoing", round(float(S[t6["meta_cond"] == "queue_ongoing", 0].mean()), 3), flush=True)
    build_eval(panel)


def build_eval(panel: str, splits=("validation", "private")):
    """Official windows of the scored splits: data 6 -> data 9 (rows before T only, slack from earlier origins)."""
    vcut = network(panel).vcut.to_numpy(np.float32)
    for split in splits:
        zz = load(panel, split)
        dates = [str(x) for x in zz["dates"]]; links = [str(x) for x in zz["links"]]; li = {l: k for k, l in enumerate(links)}
        mv2, mel2 = flat(zz["mv"]), flat(zz["melig"]).astype(bool)
        e = dict(np.load(f"{OUT}/{split}6_{panel}.npz", allow_pickle=True))
        si2 = list(e["names"]).index("slack")
        hist = pd.read_parquet(f"{RELEASE}/task2/{panel}/{split}/window_history.parquet")
        hist["link_id"] = hist.link_id.astype(str)
        feats = []
        for k, (wid, cond, o) in enumerate(zip(e["wid"], e["cond"], e["origin"])):
            if cond != "queue_ongoing":
                feats.append(np.zeros(len(SEL_NAMES), np.float32)); continue
            T = pd.Timestamp(str(o))
            i = dates.index(T.strftime("%Y-%m-%d")) * 288 + T.hour * 12 + T.minute // 5
            h = hist[hist.window_id == wid]
            kk = ((T - pd.to_datetime(h.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
            rv = np.full((HIST, len(links)), np.nan, np.float32); re = np.zeros((HIST, len(links)), bool)
            c = h.link_id.map(li).to_numpy()
            rv[HIST - kk, c] = h.speed_kmh.to_numpy(np.float32); re[HIST - kk, c] = h.is_score_eligible.to_numpy().astype(bool)
            Q, F = rows_before(i, rv, re, mv2, mel2, vcut)
            feats.append(selection_features(Q, F, float(e["X"][k][0, si2])))
        S = np.stack(feats)
        L = e["X"].shape[1]
        e["X"] = np.concatenate([e["X"], np.repeat(S[:, None, :], L, axis=1)], axis=2)
        e["names"] = np.array(list(e["names"]) + SEL_NAMES)
        np.savez(f"{OUT}/{split}9_{panel}.npz", **e)
        print(panel, split, "stable1 ongoing", int(S[:, 0].sum()), "of", int((e["cond"] == "queue_ongoing").sum()), flush=True)


if __name__ == "__main__":
    for p in (sys.argv[1:] or QUEUE_PANELS):
        build(p)
