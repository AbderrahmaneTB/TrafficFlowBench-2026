"""Task 2 v6 features: queue activity in the window's own calendar month, before the window.

The scored months run on a new scenario seed: their bottlenecks differ from train
(per-link queued share correlates 0.63-0.97 with train in March/April, 0.98-1.00 between
train months). So location knowledge has to come from the month itself: earlier days of
the same month (allowed, they are before T) and the same day before the 60-minute history.
On the first day of a month these are missing, which is the honest state of knowledge.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import WORK, QUEUE_PANELS
from t2_data import HIST
from t2_recent import queued_calendar, HALF

OUT = f"{WORK}/t2"
MONTH_NAMES = ["mq_now", "mq_h6", "mq_dayany", "mq_allday", "m_ndays", "sd_any", "sd_share", "sd_last_q"]


def month_features(Q: np.ndarray, dates: list[str], day: int, slot: int, max_days: int | None = None) -> np.ndarray:
    """(L, 8) features for a window at calendar day `day`, origin slot `slot`.
    max_days caps the month history (training windows mimic the month-start official windows)."""
    L = Q.shape[2]
    m0 = day
    while m0 > 0 and dates[m0 - 1][:7] == dates[day][:7]:
        m0 -= 1
    if max_days is not None:
        m0 = max(m0, day - max_days)
    out = []
    if m0 < day:
        for c in (slot, min(slot + 6, 287)):
            a, b = max(0, c - HALF), min(288, c + HALF + 1)
            blk = Q[m0:day, a:b]
            obs = (blk >= 0).sum(axis=(0, 1)); qd = (blk == 1).sum(axis=(0, 1))
            out.append(np.where(obs > 0, qd / np.maximum(obs, 1), np.nan))
        a, b = max(0, slot - HALF), min(288, slot + 6 + HALF + 1)
        out.append((Q[m0:day, a:b] == 1).any(axis=1).mean(axis=0))
        out.append((Q[m0:day, 60:252] == 1).any(axis=1).mean(axis=0))      # any queue 05:00-21:00
    else:
        out += [np.full(L, np.nan)] * 4
    out.append(np.full(L, day - m0, np.float64))
    # same day, before the released 60-minute history
    stop = max(slot - HIST, 0)
    blk = Q[day, :stop]
    if stop > 0:
        obs = (blk >= 0).sum(axis=0); qd = (blk == 1).sum(axis=0)
        out.append((qd > 0).astype(np.float64))
        out.append(np.where(obs > 0, qd / np.maximum(obs, 1), np.nan))
        # slots since the last queued observation today (before the history), capped
        last = np.where((blk == 1).any(axis=0), stop - 1 - np.argmax((blk == 1)[::-1], axis=0), -1)
        out.append(np.where(last >= 0, slot - last, 999).astype(np.float64))
    else:
        out += [np.zeros(L), np.full(L, np.nan), np.full(L, 999.0)]
    return np.column_stack(out).astype(np.float32)


def build_eval(panel: str, splits=("validation", "private")):
    """Official windows of the scored splits: data 3 -> data 6."""
    Q, cal = queued_calendar(panel)
    cidx = {d: k for k, d in enumerate(cal)}
    for split in splits:
        e = dict(np.load(f"{OUT}/{split}3_{panel}.npz", allow_pickle=True))
        rows = []
        for o in e["origin"]:
            T = pd.Timestamp(str(o))
            rows.append(month_features(Q, cal, cidx[T.strftime("%Y-%m-%d")], T.hour * 12 + T.minute // 5))
        e["X"] = np.concatenate([e["X"], np.stack(rows)], axis=2)
        e["names"] = np.array(list(e["names"]) + MONTH_NAMES)
        np.savez(f"{OUT}/{split}6_{panel}.npz", **e)


def build(panel: str):
    Q, cal = queued_calendar(panel)
    cidx = {d: k for k, d in enumerate(cal)}
    z = dict(np.load(f"{OUT}/train3_{panel}.npz", allow_pickle=True))
    F = np.stack([month_features(Q, cal, cidx[str(d)], int(i) % 288) for i, d in zip(z["meta_i"], z["meta_date"])])
    z["X"] = np.concatenate([z["X"], F], axis=2)
    z["names"] = np.array(list(z["names"]) + MONTH_NAMES)
    np.savez(f"{OUT}/train6_{panel}.npz", **z)
    build_eval(panel)
    print(panel, "train windows", len(F), flush=True)


if __name__ == "__main__":
    for p in (sys.argv[1:] or QUEUE_PANELS):
        build(p)
