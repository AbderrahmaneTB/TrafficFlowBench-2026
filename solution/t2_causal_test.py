"""Deletion test for Task 2 (what the award review does): for every official window, delete every released row
stamped after its origin T (masked layer of its split and of every later split, later rows of the window-history
file, later windows of the window index), rebuild that window's inputs with the production code paths
(t2_build.build_eval -> t2_build3.build_eval -> month features -> selection features; data 11 = data 9 for official
windows) and compare them with the stored inputs that the submitted forecasts were made from.

Usage: python t2_causal_test.py [panel ...]   (results appended to WORK/t2_causal_test.log)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import common
import t2_build
import t2_build3
import t2_month
import t2_sel
import t2_recent
from common import RELEASE, WORK, QUEUE_PANELS
from t2_month import month_features, MONTH_NAMES
from t2_sel import rows_before
from t2_stable import selection_features, SEL_NAMES
from t2_data import HIST

REF = f"{WORK}/t2"
TMP = tempfile.mkdtemp(prefix="t2causal_", dir=WORK)
LOG = f"{WORK}/t2_causal_test.log"
SPLIT_ORDER = {"train": 0, "validation": 1, "private": 2}

_load = common.load
_qcal = t2_recent.queued_calendar
_read_csv, _read_parquet = pd.read_csv, pd.read_parquet


def cut(panel, split, T):
    """Install truncated data sources: nothing stamped after T exists any more."""
    day = T.strftime("%Y-%m-%d")
    slot = T.hour * 12 + T.minute // 5

    def load(p, s):
        z = dict(_load(p, s))
        if SPLIT_ORDER[s] < SPLIT_ORDER[split]:
            return z
        dates = [str(x) for x in z["dates"]]
        D = len(dates)
        keep = np.zeros((D, 288), bool)
        if SPLIT_ORDER[s] == SPLIT_ORDER[split]:
            d = dates.index(day)
            keep[:d] = True
            keep[d, :slot + 1] = True
        for k in ("mv", "mq", "mo", "v", "q", "o"):
            if k in z:
                a = z[k].astype(np.float32).copy(); a[~keep] = np.nan; z[k] = a
        for k in ("melig", "elig"):
            if k in z:
                a = z[k].copy(); a[~keep] = 0; z[k] = a
        return z

    def qcal(p):
        Q, dates = _qcal(p)
        Q = Q.copy()
        d = dates.index(day)
        Q[d + 1:] = -1
        Q[d, slot + 1:] = -1
        return Q, dates

    def read_csv(path, *a, **k):
        df = _read_csv(path, *a, **k)
        if str(path).replace("\\", "/").endswith(f"task2/{panel}/{split}/window_index.csv"):
            df = df[pd.to_datetime(df.forecast_origin, utc=True) <= T].reset_index(drop=True)
        return df

    def read_parquet(path, *a, **k):
        df = _read_parquet(path, *a, **k)
        if str(path).replace("\\", "/").endswith(f"task2/{panel}/{split}/window_history.parquet"):
            df = df[pd.to_datetime(df.timestamp, utc=True) <= T].reset_index(drop=True)
        return df

    for m in (t2_build, t2_build3, t2_sel):
        m.load = load
    t2_build3.queued_calendar = qcal
    pd.read_csv, pd.read_parquet = read_csv, read_parquet
    for m in (t2_build, t2_build3, t2_sel):
        m.OUT = TMP
    return load, qcal


def restore():
    for m in (t2_build, t2_build3, t2_sel):
        m.load = _load
        m.OUT = REF
    t2_build3.queued_calendar = _qcal
    pd.read_csv, pd.read_parquet = _read_csv, _read_parquet


def rebuild(panel, split, T):
    """The window inputs of every window with origin <= T, from truncated data only."""
    load, qcal = cut(panel, split, T)
    try:
        head1 = np.load(f"{REF}/train_{panel}.npz")["head"]
        t2_build.build_eval(panel, split, head1)
        head3 = np.load(f"{REF}/train3_{panel}.npz", allow_pickle=True)["head"]
        t2_build3.build_eval(panel, split, head3)
        # month features (t2_month.build, official part)
        Q, cal = qcal(panel)
        cidx = {d: k for k, d in enumerate(cal)}
        e = dict(np.load(f"{TMP}/{split}3_{panel}.npz", allow_pickle=True))
        rows = []
        for o in e["origin"]:
            To = pd.Timestamp(str(o))
            rows.append(month_features(Q, cal, cidx[To.strftime("%Y-%m-%d")], To.hour * 12 + To.minute // 5))
        e["X"] = np.concatenate([e["X"], np.stack(rows)], axis=2)
        e["names"] = np.array(list(e["names"]) + MONTH_NAMES)
        # selection features (t2_sel.build, official part)
        vcut = common.network(panel).vcut.to_numpy(np.float32)
        zz = load(panel, split)
        dates = [str(x) for x in zz["dates"]]; links = [str(x) for x in zz["links"]]; li = {l: k for k, l in enumerate(links)}
        mv2, mel2 = t2_build.flat(zz["mv"]), t2_build.flat(zz["melig"]).astype(bool)
        si2 = list(e["names"]).index("slack")
        hist = pd.read_parquet(f"{RELEASE}/task2/{panel}/{split}/window_history.parquet")
        hist["link_id"] = hist.link_id.astype(str)
        feats = []
        for k, (wid, cond, o) in enumerate(zip(e["wid"], e["cond"], e["origin"])):
            if cond != "queue_ongoing":
                feats.append(np.zeros(len(SEL_NAMES), np.float32)); continue
            To = pd.Timestamp(str(o))
            i = dates.index(To.strftime("%Y-%m-%d")) * 288 + To.hour * 12 + To.minute // 5
            h = hist[hist.window_id == wid]
            kk = ((To - pd.to_datetime(h.timestamp, utc=True)).dt.total_seconds() // 300).astype(int).to_numpy()
            rv = np.full((HIST, len(links)), np.nan, np.float32); re = np.zeros((HIST, len(links)), bool)
            c = h.link_id.map(li).to_numpy()
            rv[HIST - kk, c] = h.speed_kmh.to_numpy(np.float32); re[HIST - kk, c] = h.is_score_eligible.to_numpy().astype(bool)
            Qb, Fb = rows_before(i, rv, re, mv2, mel2, vcut)
            feats.append(selection_features(Qb, Fb, float(e["X"][k][0, si2])))
        S = np.stack(feats)
        L = e["X"].shape[1]
        e["X"] = np.concatenate([e["X"], np.repeat(S[:, None, :], L, axis=1)], axis=2)
        return e
    finally:
        restore()


def main():
    panels = sys.argv[1:] or QUEUE_PANELS
    for panel in panels:
        for split in ("validation", "private"):
            ref = np.load(f"{REF}/{split}11_{panel}.npz", allow_pickle=True)
            rX = {str(w): x for w, x in zip(ref["wid"], ref["X"])}
            wi = _read_csv(f"{RELEASE}/task2/{panel}/{split}/window_index.csv")
            n_ok = 0; bad = []
            for wid, o in zip(wi.window_id.astype(str), wi.forecast_origin):
                T = pd.Timestamp(o)
                T = T.tz_localize("UTC") if T.tzinfo is None else T
                e = rebuild(panel, split, T)
                got = {str(w): x for w, x in zip(e["wid"], e["X"])}
                a, b = got[wid], rX[wid]
                same = a.shape == b.shape and np.array_equal(np.nan_to_num(a, nan=-7.7e7), np.nan_to_num(b, nan=-7.7e7))
                if same:
                    n_ok += 1
                else:
                    diffc = np.flatnonzero(~np.isclose(np.nan_to_num(a, nan=-7.7e7), np.nan_to_num(b, nan=-7.7e7)).all(axis=0)) \
                        if a.shape == b.shape else []
                    bad.append((wid, [str(ref["names"][j]) for j in diffc][:8]))
            line = f"{panel} {split}: {n_ok}/{len(wi)} windows identical after deleting rows > T" + (f"; DIFF {bad}" if bad else "")
            print(line, flush=True)
            with open(LOG, "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
