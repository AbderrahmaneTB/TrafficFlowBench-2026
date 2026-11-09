"""Build Task 2 training windows (train split) and inference windows (validation/private)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from common import RELEASE, WORK, QUEUE_PANELS, load, network
from t1_features import profiles
from t2_data import HIST, CTX, PanelFeatures, select_windows

OUT = f"{WORK}/t2"
os.makedirs(OUT, exist_ok=True)
NROW = CTX + HIST + 1
CAND = 30  # ongoing: links farther than this from any queue at T are predicted free


def flat(a):
    D, T, L = a.shape
    return a.reshape(D * T, L)


def head_rate_from(windows, Qeli, L):
    """Share of onset events in which each link is queued at T+30."""
    cnt = np.zeros(L); n = 0
    for i, cond in windows:
        if cond == "queue_onset":
            cnt += Qeli[i + 6]; n += 1
    return (cnt / max(n, 1)).astype(np.float32)


def context(i, raw, masked):
    """Rows i-NROW+1..i: masked context, raw history rows i-12..i-1, masked row i."""
    lo = i - NROW + 1
    out = []
    for k, (rv, mv) in enumerate(zip(raw, masked)):
        a = mv[lo:i + 1].copy()
        a[CTX:CTX + HIST] = rv[i - HIST:i]
        out.append(a)
    return out


def build_train(panel: str):
    z = load(panel, "train")
    net = network(panel)
    dates = [str(x) for x in z["dates"]]
    v, q, o = flat(z["v"]), flat(z["q"]), flat(z["o"])
    mv, mq, mo = flat(z["mv"]), flat(z["mq"]), flat(z["mo"])
    el = flat(z["elig"]).astype(bool)
    vcut = net.vcut.to_numpy(np.float32)
    D = len(dates)
    wins = set()
    for sp in (72, 48, 60, 84, 96, 120):
        wins |= set(select_windows(v, el, vcut, spacing=sp))
    wins = sorted(wins)
    Q = (np.isfinite(v) & (v <= vcut) & el)
    day_of = np.repeat(np.arange(D), 288)
    month = np.array([dates[d][:7] for d in day_of])
    hold = month == "2031-02"
    fit_wins = [(i, c) for i, c in wins if not hold[i]]
    head = head_rate_from(fit_wins, Q, Q.shape[1])
    prof = profiles(panel)
    pf = PanelFeatures(panel, prof["qu"], head)
    dows = pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()
    rows_X, rows_y, rows_m, meta = [], [], [], []
    names = None
    for wi, (i, cond) in enumerate(wins):
        if i - NROW + 1 < 0:
            continue
        V, Qf, O = context(i, (v, q, o), (mv, mq, mo))
        X, names = pf.features(V, Qf, O, i % 288, dows[day_of[i]], cond)
        y = Q[i + 1:i + 7]; m = el[i + 1:i + 7] & np.isfinite(v[i + 1:i + 7])
        rows_X.append(X.astype(np.float32)); rows_y.append(y); rows_m.append(m)
        meta.append((wi, i, cond, bool(hold[i]), dates[day_of[i]]))
    np.savez(f"{OUT}/train_{panel}.npz", X=np.stack(rows_X), y=np.stack(rows_y), m=np.stack(rows_m),
             names=np.array(names), meta_i=np.array([x[1] for x in meta]),
             meta_cond=np.array([x[2] for x in meta]), meta_hold=np.array([x[3] for x in meta]),
             meta_date=np.array([x[4] for x in meta]), head=head)
    nc = pd.Series([x[2] for x in meta]).value_counts().to_dict()
    print(panel, "windows", len(meta), nc, "features", len(names), flush=True)


def build_eval(panel: str, split: str, head: np.ndarray):
    z = load(panel, split)
    net = network(panel)
    dates = [str(x) for x in z["dates"]]
    links = [str(x) for x in z["links"]]
    mv, mq, mo = flat(z["mv"]), flat(z["mq"]), flat(z["mo"])
    wi = pd.read_csv(f"{RELEASE}/task2/{panel}/{split}/window_index.csv")
    hist = pd.read_parquet(f"{RELEASE}/task2/{panel}/{split}/window_history.parquet")
    hist["link_id"] = hist.link_id.astype(str)
    prof = profiles(panel)
    pf = PanelFeatures(panel, prof["qu"], head)
    li = {l: k for k, l in enumerate(links)}
    Xs, meta = [], []
    names = None
    for w in wi.itertuples():
        T = pd.Timestamp(w.forecast_origin)
        d = dates.index(w.date)
        i = d * 288 + T.hour * 12 + T.minute // 5
        lo = i - NROW + 1
        rv = [np.full((HIST, len(links)), np.nan, np.float32) for _ in range(3)]
        h = hist[hist.window_id == w.window_id]
        ts = pd.to_datetime(h.timestamp, utc=True)
        k = ((T - ts).dt.total_seconds() // 300).astype(int).to_numpy()   # 1..12 steps before T
        assert (k >= 1).all() and (k <= HIST).all(), "history must be strictly before T"
        r = HIST - k
        c = h.link_id.map(li).to_numpy()
        for arr, col in zip(rv, ("speed_kmh", "flow_vph", "occupancy")):
            arr[r, c] = h[col].to_numpy(np.float32)
        V, Qf, O = [], [], []
        for mat, raw, dst in ((mv, rv[0], V), (mq, rv[1], Qf), (mo, rv[2], O)):
            a = np.full((NROW, len(links)), np.nan, np.float32)
            src_lo = max(lo, 0)
            a[src_lo - lo:] = mat[src_lo:i + 1]          # rows up to and including T only
            a[CTX:CTX + HIST] = raw
            dst.append(a)
        X, names = pf.features(V[0], Qf[0], O[0], i % 288, T.dayofweek, w.condition)
        Xs.append(X.astype(np.float32)); meta.append((w.window_id, w.condition, w.forecast_origin))
    np.savez(f"{OUT}/{split}_{panel}.npz", X=np.stack(Xs), names=np.array(names),
             wid=np.array([m[0] for m in meta]), cond=np.array([m[1] for m in meta]),
             origin=np.array([m[2] for m in meta]), links=np.array(links))
    print(panel, split, len(meta), flush=True)


if __name__ == "__main__":
    panels = sys.argv[1:] or QUEUE_PANELS
    for p in panels:
        build_train(p)
        head = np.load(f"{OUT}/train_{p}.npz")["head"]
        for s in ("validation", "private"):
            build_eval(p, s, head)
