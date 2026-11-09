"""Task 2 v5 datasets: the v3 windows, with leave-one-month-out location priors.

In v3 the queued-share profile and the onset head rate of a training window were computed
over every train month, its own month and its own onset included. A month always agrees with
its own average, so the model learned to trust those priors more than they deserve on March
and April, where bottlenecks switch on and off. Here every training window sees priors from
the other train months only, which is exactly the situation of a scored month.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import WORK, QUEUE_PANELS, load, network
from t1_features import profiles
from t2_build import context, flat, NROW
from t2_build3 import EXTRA, chains
from t2_data import PanelFeatures
from t2_recent import queued_calendar, recent_features
from t2_underlying import underlying_queue

OUT = f"{WORK}/t2"


def build_train(panel: str):
    z = load(panel, "train")
    net = network(panel)
    dates = [str(x) for x in z["dates"]]
    D = len(dates)
    vcut = net.vcut.to_numpy(np.float32)
    # queued share by (slot, link) per month, from the unmasked layer (as t1_features.profiles)
    v3d = np.where(z["elig"].astype(bool), z["v"], np.nan)
    fin = np.isfinite(v3d); qd = fin & (v3d <= vcut)
    mon_day = np.array([d[:7] for d in dates])
    months = sorted(set(mon_day))
    S = {m: qd[mon_day == m].sum(0) for m in months}; C = {m: fin[mon_day == m].sum(0) for m in months}
    St, Ct = sum(S.values()), sum(C.values())
    prof_lomo = {m: ((St - S[m]) / np.maximum(Ct - C[m], 1)).astype(np.float32) for m in months}
    del v3d, fin, qd
    v, q, o = flat(z["v"]), flat(z["q"]), flat(z["o"])
    mv, mq, mo = flat(z["mv"]), flat(z["mq"]), flat(z["mo"])
    el = flat(z["elig"]).astype(bool)
    U = underlying_queue(v, vcut)
    wins = chains(v, el, vcut, U, D)
    Qcal, cal_dates = queued_calendar(panel)
    cidx = {d: k for k, d in enumerate(cal_dates)}
    month = np.array([dates[i // 288][:7] for i in range(len(v))])
    # onset head rate per month, then leave-one-month-out
    Hc = {m: np.zeros(v.shape[1]) for m in months}; Hn = {m: 0 for m in months}
    for i, c, *_ in wins:
        if c == "queue_onset":
            Hc[month[i]] += U[i + 6] & el[i + 6]; Hn[month[i]] += 1
    Ht, Nt = sum(Hc.values()), sum(Hn.values())
    head_lomo = {m: ((Ht - Hc[m]) / max(Nt - Hn[m], 1)).astype(np.float32) for m in months}
    head_all = (Ht / max(Nt, 1)).astype(np.float32)
    pfs = {m: PanelFeatures(panel, prof_lomo[m], head_lomo[m]) for m in months}
    dows = pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()
    Xs, ys, ms, meta = [], [], [], []
    names = None
    for i, cond, slack, since, n_on in wins:
        if i - NROW + 1 < 0:
            continue
        V, Qf, O = context(i, (v, q, o), (mv, mq, mo))
        X, names = pfs[month[i]].features(V, Qf, O, i % 288, dows[i // 288], cond)
        r = recent_features(Qcal, cidx[dates[i // 288]], i % 288)
        ex = np.column_stack([np.full(len(X), slack), np.full(len(X), min(since, 999)), np.full(len(X), n_on),
                              *r]).astype(np.float32)
        Xs.append(np.column_stack([X, ex]).astype(np.float32))
        ys.append(U[i + 1:i + 7]); ms.append(el[i + 1:i + 7] & np.isfinite(v[i + 1:i + 7]))
        meta.append((i, cond, month[i] == "2031-02", dates[i // 288]))
    np.savez(f"{OUT}/train5_{panel}.npz", X=np.stack(Xs), y=np.stack(ys), m=np.stack(ms),
             names=np.array(names + EXTRA), meta_i=np.array([x[0] for x in meta]),
             meta_cond=np.array([x[1] for x in meta]), meta_hold=np.array([x[2] for x in meta]),
             meta_date=np.array([x[3] for x in meta]), head=head_all)
    print(panel, "windows", len(meta), flush=True)
    return head_all


def build_eval(panel: str, split: str, head: np.ndarray):
    """v3 eval windows with the all-month head rate (the scored month is never a train month)."""
    e = np.load(f"{OUT}/{split}3_{panel}.npz", allow_pickle=True)
    names = list(e["names"])
    X = e["X"].copy()
    X[:, :, names.index("head_rate")] = head[None, :]
    prof = profiles(panel)["qu"]
    assert prof.shape[1] == X.shape[1]
    np.savez(f"{OUT}/{split}5_{panel}.npz", X=X, names=e["names"], wid=e["wid"], cond=e["cond"],
             origin=e["origin"], links=e["links"])
    print(panel, split, "eval windows", len(X), flush=True)


if __name__ == "__main__":
    for p in (sys.argv[1:] or QUEUE_PANELS):
        h = build_train(p)
        for s in ("validation", "private"):
            build_eval(p, s, h)
