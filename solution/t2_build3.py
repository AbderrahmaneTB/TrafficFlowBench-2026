"""Task 2 v3 datasets: windows cut with the reverse-engineered official selector.

The selector (reproduces 73/80 official train origins exactly, 80/80 within 15 minutes):
onset windows first (earliest qualifying origin, queue first at T+30 on a scored cell,
360-minute spacing), then ongoing windows spaced 360 minutes from every chosen window,
with persistence IoU <= 0.9 measured on the underlying state.

Training windows are the union of selector runs started on every train day, so they follow
the official distribution. Labels use the estimated underlying queue state on scored cells.
Features use only data at or before T; `slack` is derived from earlier window origins only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from common import RELEASE, WORK, QUEUE_PANELS, load, network
from t1_features import profiles
from t2_build import context, flat, NROW
from t2_data import HIST, SPACING, PanelFeatures, select_official
from t2_recent import queued_calendar, recent_features
from t2_underlying import underlying_queue

OUT = f"{WORK}/t2"
EXTRA = ["slack", "since_prev", "n_prev_onset", "rq7_now", "rq7_h6", "rq7_dayany"]


def chains(v, el, vcut, U, days, span=14):
    """Union of official-style selections started at each day; slack per window from its chain."""
    N = v.shape[0]
    seen = {}
    for d0 in range(0, days - 2):
        start = d0 * 288 + HIST
        stop = min(N - 6, (d0 + span) * 288)
        w = select_official(v, el, vcut, U, start=start, stop=stop)
        prev = []
        for i, cond in w:
            earlier = [j for j, c in prev if j < i and (cond == "queue_ongoing" or c == "queue_onset")]
            allowed = max([j + SPACING for j in earlier], default=start)
            slack = i - allowed
            since = i - max(earlier) if earlier else 999
            n_on = sum(1 for j, c in prev if j < i and c == "queue_onset")
            prev.append((i, cond))
            if (i, cond) not in seen:
                seen[(i, cond)] = (slack, since, n_on)
    return sorted((i, c, *seen[(i, c)]) for i, c in seen)


def build_train(panel: str):
    z = load(panel, "train")
    net = network(panel)
    dates = [str(x) for x in z["dates"]]
    D = len(dates)
    v, q, o = flat(z["v"]), flat(z["q"]), flat(z["o"])
    mv, mq, mo = flat(z["mv"]), flat(z["mq"]), flat(z["mo"])
    el = flat(z["elig"]).astype(bool)
    vcut = net.vcut.to_numpy(np.float32)
    U = underlying_queue(v, vcut)
    wins = chains(v, el, vcut, U, D)
    Qcal, cal_dates = queued_calendar(panel)
    cidx = {d: k for k, d in enumerate(cal_dates)}
    month = np.array([dates[i // 288][:7] for i in range(len(v))])
    prof = profiles(panel)
    # head rate from June-January onset windows (underlying label at T+30)
    head = np.zeros(v.shape[1]); n = 0
    for i, c, *_ in wins:
        if c == "queue_onset" and month[i] != "2031-02":
            head += U[i + 6] & el[i + 6]; n += 1
    head = (head / max(n, 1)).astype(np.float32)
    pf = PanelFeatures(panel, prof["qu"], head)
    dows = pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()
    Xs, ys, ms, meta = [], [], [], []
    names = None
    for i, cond, slack, since, n_on in wins:
        if i - NROW + 1 < 0:
            continue
        V, Qf, O = context(i, (v, q, o), (mv, mq, mo))
        X, names = pf.features(V, Qf, O, i % 288, dows[i // 288], cond)
        r = recent_features(Qcal, cidx[dates[i // 288]], i % 288)
        ex = np.column_stack([np.full(len(X), slack), np.full(len(X), min(since, 999)), np.full(len(X), n_on),
                              *r]).astype(np.float32)
        Xs.append(np.column_stack([X, ex]).astype(np.float32))
        ys.append(U[i + 1:i + 7]); ms.append(el[i + 1:i + 7] & np.isfinite(v[i + 1:i + 7]))
        meta.append((i, cond, month[i] == "2031-02", dates[i // 288]))
    np.savez(f"{OUT}/train3_{panel}.npz", X=np.stack(Xs), y=np.stack(ys), m=np.stack(ms),
             names=np.array(names + EXTRA), meta_i=np.array([x[0] for x in meta]),
             meta_cond=np.array([x[1] for x in meta]), meta_hold=np.array([x[2] for x in meta]),
             meta_date=np.array([x[3] for x in meta]), head=head)
    print(panel, "windows", len(meta), pd.Series([x[1] for x in meta]).value_counts().to_dict(), flush=True)
    return head


def build_eval(panel: str, split: str, head: np.ndarray):
    """Official windows; slack from window_index origins strictly before T (allowed: <= T)."""
    e = np.load(f"{OUT}/{split}_{panel}.npz", allow_pickle=True)
    wi = pd.read_csv(f"{RELEASE}/task2/{panel}/{split}/window_index.csv")
    org = pd.to_datetime(wi.forecast_origin, utc=True)
    t0 = pd.Timestamp(wi.date.min() + "T00:00:00Z") + pd.Timedelta(minutes=5 * HIST)
    Qcal, cal_dates = queued_calendar(panel)
    cidx = {d: k for k, d in enumerate(cal_dates)}
    # rebuild base features with the v3 head rate
    z = load(panel, split)
    from t2_build import build_eval as _be  # noqa: F401  (base features already cached)
    X0 = e["X"]; names0 = list(e["names"])
    hi = names0.index("head_rate")
    out = []
    for k, (wid, cond, o) in enumerate(zip(e["wid"], e["cond"], e["origin"])):
        T = pd.Timestamp(str(o))
        same = (wi.condition == "queue_onset").to_numpy() if cond == "queue_onset" else np.ones(len(wi), bool)
        earlier = org[(org < T).to_numpy() & same]
        allowed = max([x + pd.Timedelta(minutes=360) for x in earlier], default=t0)
        slack = (T - allowed).total_seconds() / 300
        since = (T - earlier.max()).total_seconds() / 300 if len(earlier) else 999
        n_on = int(((org < T) & (wi.condition == "queue_onset")).sum())
        r = recent_features(Qcal, cidx[T.strftime("%Y-%m-%d")], T.hour * 12 + T.minute // 5)
        X = X0[k].copy(); X[:, hi] = head
        ex = np.column_stack([np.full(len(X), slack), np.full(len(X), min(since, 999)), np.full(len(X), n_on), *r])
        out.append(np.column_stack([X, ex]).astype(np.float32))
    np.savez(f"{OUT}/{split}3_{panel}.npz", X=np.stack(out), names=np.array(names0 + EXTRA),
             wid=e["wid"], cond=e["cond"], origin=e["origin"], links=e["links"])
    print(panel, split, "eval windows", len(out), flush=True)


if __name__ == "__main__":
    for p in (sys.argv[1:] or QUEUE_PANELS):
        h = build_train(p)
        for s in ("validation", "private"):
            build_eval(p, s, h)
