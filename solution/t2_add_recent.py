"""Attach causal recent-activity features to the Task 2 window datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import WORK, QUEUE_PANELS
from t2_recent import queued_calendar, recent_features

OUT = f"{WORK}/t2"
NAMES = ["rq7_now", "rq7_h6", "rq7_dayany"]

if __name__ == "__main__":
    for p in QUEUE_PANELS:
        Q, dates = queued_calendar(p)
        didx = {d: i for i, d in enumerate(dates)}
        z = np.load(f"{OUT}/train_{p}.npz", allow_pickle=True)
        feats = []
        for i, date in zip(z["meta_i"], z["meta_date"]):
            feats.append(np.stack(recent_features(Q, didx[str(date)], int(i) % 288), axis=1))
        np.savez(f"{OUT}/recent_train_{p}.npz", R=np.stack(feats))
        for s in ("validation", "private"):
            e = np.load(f"{OUT}/{s}_{p}.npz", allow_pickle=True)
            feats = []
            for o in e["origin"]:
                T = pd.Timestamp(str(o))
                feats.append(np.stack(recent_features(Q, didx[T.strftime("%Y-%m-%d")], T.hour * 12 + T.minute // 5), axis=1))
            np.savez(f"{OUT}/recent_{s}_{p}.npz", R=np.stack(feats))
        print(p, "done", flush=True)
