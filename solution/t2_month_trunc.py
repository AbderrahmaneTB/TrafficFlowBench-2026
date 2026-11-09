"""Task 2 data "11": data "9" with the month-own features of TRAINING windows recomputed from a
month history cut to a random 0-6 days.

Official windows sit on days 1-5 of their month (onset) or 1-12 (ongoing), so their month-own
history is 0-4 days long; training windows spread over whole months saw up to 30 days and the
model learned to read month features of a quality it never gets at inference. Held-out February
windows keep their true month history, and validation/private windows are unchanged copies of "9".
Trained together with prior shift (t2_train --prior-shift): the train-period location priors are
moved to wrong places in half the windows while the month-own features stay truthful, which is the
situation of a new scenario month.
"""
from __future__ import annotations

import shutil
import sys

import numpy as np

from common import WORK, QUEUE_PANELS
from t2_month import MONTH_NAMES, month_features
from t2_recent import queued_calendar

OUT = f"{WORK}/t2"
MAX_K = 6


def build(panel: str, seed: int = 0):
    Q, cal = queued_calendar(panel)
    cidx = {d: k for k, d in enumerate(cal)}
    z = dict(np.load(f"{OUT}/train9_{panel}.npz", allow_pickle=True))
    names = list(z["names"])
    cols = [names.index(n) for n in MONTH_NAMES]
    rng = np.random.default_rng(seed + QUEUE_PANELS.index(panel))
    X = z["X"]
    for w, (i, d, hold) in enumerate(zip(z["meta_i"], z["meta_date"], z["meta_hold"])):
        if hold:
            continue
        X[w][:, cols] = month_features(Q, cal, cidx[str(d)], int(i) % 288, max_days=int(rng.integers(0, MAX_K + 1)))
    z["X"] = X
    np.savez(f"{OUT}/train11_{panel}.npz", **z)
    for split in ("validation", "private"):
        shutil.copy(f"{OUT}/{split}9_{panel}.npz", f"{OUT}/{split}11_{panel}.npz")
    print(panel, "train windows", len(X), flush=True)


if __name__ == "__main__":
    for p in (sys.argv[1:] or QUEUE_PANELS):
        build(p)
