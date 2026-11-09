"""Task 2 windows emulated in a scored month (March), for choosing and fitting APRIL models only.

The scored months run on new scenarios, unlike the nine near-identical train months. March is the only
other scenario available, and organizers allow March data for April windows (it is all before every
April origin). For March's own windows it would be data after T, so nothing built here may touch the
validation-split predictions.

Windows are cut with the reverse-engineered selector on the masked March layer (the raw layer is not
released): the history therefore carries the Task 1 blanks on top of detector gaps, and labels are the
gap-filled underlying estimate on observed horizon cells only. Same feature layout as data "9".
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import WORK, QUEUE_PANELS, load, network
from t1_features import profiles
from t2_build import context, flat, NROW
from t2_build3 import EXTRA, chains
from t2_data import HIST, PanelFeatures
from t2_month import MONTH_NAMES, month_features
from t2_recent import queued_calendar, recent_features
from t2_sel import rows_before
from t2_stable import SEL_NAMES, selection_features
from t2_underlying import fill_time, underlying_queue
from t1_features import target_index

OUT = f"{WORK}/t2"


def build(panel: str, split: str = "validation"):
    z = load(panel, split)
    net = network(panel)
    vcut = net.vcut.to_numpy(np.float32)
    dates = [str(x) for x in z["dates"]]
    D = len(dates)
    mv, mq, mo = flat(z["mv"]), flat(z["mq"]), flat(z["mo"])
    mel = flat(z["melig"]).astype(bool)
    U = underlying_queue(mv, vcut)
    # the selector sees the raw layer; here only the Task 1 targets are filled (time interpolation)
    # so that coverage and the visible-queue condition behave as they would on raw data
    idx = target_index(panel, split, dates, [str(x) for x in z["links"]])
    flat_t = idx["d"].astype(np.int64) * 288 + idx["t"].astype(np.int64)
    fill = {}
    for k, a in (("v", mv), ("q", mq), ("o", mo)):
        b = a.copy(); b[flat_t, idx["l"]] = fill_time(a)[flat_t, idx["l"]]; fill[k] = b
    v_sel = fill["v"]
    wins = chains(v_sel, mel, vcut, U, D)
    head = np.load(f"{OUT}/train3_{panel}.npz", allow_pickle=True)["head"]
    pf = PanelFeatures(panel, profiles(panel)["qu"], head)
    Qcal, cal = queued_calendar(panel)
    cidx = {d: k for k, d in enumerate(cal)}
    dows = pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()
    Xs, ys, ms, meta = [], [], [], []
    names = None
    for i, cond, slack, since, n_on in wins:
        if i - NROW + 1 < 0:
            continue
        # released history of a scored window is raw: use the target-filled rows there, masked rows before
        V, Qf, O = context(i, (fill["v"], fill["q"], fill["o"]), (mv, mq, mo))
        X, names = pf.features(V, Qf, O, i % 288, dows[i // 288], cond)
        day = cidx[dates[i // 288]]
        r = recent_features(Qcal, day, i % 288)
        ex = np.column_stack([np.full(len(X), slack), np.full(len(X), min(since, 999)), np.full(len(X), n_on), *r])
        mf = month_features(Qcal, cal, day, i % 288)
        if cond == "queue_ongoing":
            Qb, Fb = rows_before(i, v_sel[i - HIST:i], mel[i - HIST:i], mv, mel, vcut)
            sel = selection_features(Qb, Fb, float(slack))
        else:
            sel = np.zeros(len(SEL_NAMES), np.float32)
        Xs.append(np.column_stack([X, ex, mf, np.repeat(sel[None], len(X), axis=0)]).astype(np.float32))
        # label cells = the cells the selection saw (raw-like: Task 1 targets filled), as the official
        # labels cover every raw-observed horizon cell; masking the targets out left onset windows
        # whose T+30 queue sat only on target cells with an empty truth
        ys.append(U[i + 1:i + 7]); ms.append(mel[i + 1:i + 7] & np.isfinite(v_sel[i + 1:i + 7]))
        meta.append((i, cond, dates[i // 288]))
    np.savez(f"{OUT}/{split}M_{panel}.npz", X=np.stack(Xs), y=np.stack(ys), m=np.stack(ms),
             names=np.array(names + EXTRA + MONTH_NAMES + SEL_NAMES), meta_i=np.array([x[0] for x in meta]),
             meta_cond=np.array([x[1] for x in meta]), meta_date=np.array([x[2] for x in meta]))
    c = pd.Series([x[1] for x in meta]).value_counts().to_dict()
    print(panel, split, "emulated windows", len(meta), c, flush=True)


if __name__ == "__main__":
    for p in (sys.argv[1:] or QUEUE_PANELS):
        build(p)
