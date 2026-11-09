"""Task 2 only: rebuild the queue forecasts of the scored splits from the data under TFB_DATA and write them in the
official per-task format (window_id, timestamp, link_id, queue_pred), ready for the repository's merge_submissions.py.

Made for the award review's two Task 2 reruns (organizer post of Nov 4, 2026): the release with every row stamped
after each window's origin removed, and a newly generated month with the same file layout. Everything a forecast
learns from is fitted on the TRAIN split (training windows, location priors, LightGBM models), so it is reused from
TFB_WORK; every input of a scored window is rebuilt from what TFB_DATA holds now:

  TFB_DATA=<release> TFB_WORK=<work dir of run_all.py> python run_task2.py --spec specs/final_a.json
      [--out queue_submission.csv] [--into full_submission.csv] [--keep-cache]

  1. cache of the scored splits of the queue panels (build_cache.py)          ~2 min per panel
  2. queued calendar (t2_recent.queued_calendar: train + scored splits)        rebuilt
  3. official-window inputs: t2_build -> t2_build3 -> t2_month -> t2_sel -> data 11 (= data 9 for official windows)
  4. forecasts: make_final.queue_from_spec (the spec's models and decision rules, full models)
  5. queue_submission.csv in the official per-task format (rows of submission_key.csv with task == queue);
     --into also writes <full_submission>_t2.csv = that file with its queue_pred column replaced.

The train-split stages (run_all.py stages 1, 3 and 4: train cache, t2/train*_<panel>.npz, the models of the spec) must
exist in TFB_WORK. A window only ever reads rows stamped at or before its origin (README, "Task 2 and time"), so
deleting later rows from TFB_DATA leaves every forecast unchanged (t2_causal_test.py checks this window by window).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from common import CACHE, RELEASE, WORK, QUEUE_PANELS

OUT = f"{WORK}/t2"
# the scored splits found under TFB_DATA: both for the release, possibly one for a newly generated month
SCORED = tuple(s for s in ("validation", "private")
               if all(os.path.exists(f"{RELEASE}/task2/{p}/{s}/window_index.csv") for p in QUEUE_PANELS))


def check_prerequisites(spec: dict):
    missing = []
    for p in QUEUE_PANELS:
        for f in (f"{CACHE}/{p}_train.npz", f"{OUT}/train_{p}.npz", f"{OUT}/train3_{p}.npz"):
            if not os.path.exists(f):
                missing.append(f)
    tags = sorted({t for s in spec.values() for c in ("onset", "ongoing") for t in s[c]})
    for t in tags:
        cfg = json.load(open(f"{OUT}/config_{t}.json"))
        if cfg.get("data") != "11":
            raise SystemExit(f"run_task2.py rebuilds data 11 only; tag {t} uses data {cfg.get('data')}")
        mt = cfg.get("model_tag", t)
        for kind in ("ongoing", "onset"):
            f = f"{OUT}/{kind}_{mt}_full.txt"
            if not os.path.exists(f):
                missing.append(f)
    for s in spec.values():
        if "remote" in s:                              # optional remote-queue additions (t2_remote.py)
            from t2_remote import model_files
            missing += [f for f in model_files(s["remote"]) if not os.path.exists(f)]
    if missing:
        raise SystemExit("missing train-split outputs (run run_all.py first):\n  " + "\n  ".join(missing))
    return tags


def rebuild_inputs(keep_cache: bool):
    import build_cache
    import t2_build
    import t2_build3
    import t2_month
    import t2_sel
    for p in QUEUE_PANELS:
        t0 = time.time()
        if not keep_cache:
            build_cache.build(p, SCORED)
        qc = f"{OUT}/qcal_{p}.npz"
        if os.path.exists(qc):
            os.remove(qc)                              # stitched calendar of train + scored splits: rebuilt below
        head = np.load(f"{OUT}/train_{p}.npz")["head"]
        head3 = np.load(f"{OUT}/train3_{p}.npz", allow_pickle=True)["head"]
        for s in SCORED:
            t2_build.build_eval(p, s, head)            # -> {s}_{p}.npz   (window history + rows <= T)
            t2_build3.build_eval(p, s, head3)          # -> {s}3_{p}.npz  (+ slack from earlier origins, rq7_*)
        t2_month.build_eval(p, SCORED)                 # -> {s}6_{p}.npz  (+ earlier days of the month)
        t2_sel.build_eval(p, SCORED)                   # -> {s}9_{p}.npz  (+ selection features, rows before T)
        for s in SCORED:                               # data 11 = data 9 for official windows (t2_month_trunc.py)
            shutil.copy(f"{OUT}/{s}9_{p}.npz", f"{OUT}/{s}11_{p}.npz")
        print(p, f"inputs rebuilt in {time.time() - t0:.0f} s", flush=True)


def queue_rows(queue: dict) -> pd.DataFrame:
    """The queue rows of submission_key.csv with our 0 / 1 forecast."""
    links, origin = {}, {}
    for p in QUEUE_PANELS:
        links[p] = {str(x): i for i, x in enumerate(np.load(f"{CACHE}/{p}_{SCORED[0]}.npz", allow_pickle=True)["links"])}
        for s in SCORED:
            wi = pd.read_csv(f"{RELEASE}/task2/{p}/{s}/window_index.csv")
            origin.update(dict(zip(wi.window_id.astype(str), pd.to_datetime(wi.forecast_origin, utc=True))))
    parts = []
    reader = pd.read_csv(f"{RELEASE}/submission_key.csv", dtype=str, chunksize=1_000_000,
                         usecols=["submission_id", "task", "panel", "timestamp", "link_id", "window_id"])
    for k in reader:
        k = k[(k.task == "queue") & k.window_id.isin(origin.keys())]     # windows of the splits present
        if len(k):
            parts.append(k)
    k = pd.concat(parts, ignore_index=True)
    ts = pd.to_datetime(k.timestamp, utc=True)
    h = ((ts - k.window_id.map(origin)).dt.total_seconds() // 300).astype(int).to_numpy()
    assert ((h >= 1) & (h <= 6)).all(), "queue rows must lie at T+5 ... T+30"
    li = np.array([links[p][l] for p, l in zip(k.panel, k.link_id)])
    k["queue_pred"] = [int(queue[w][hh - 1, ll]) for w, hh, ll in zip(k.window_id, h, li)]
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="per-split Task 2 settings (the spec of the final submission)")
    ap.add_argument("--out", default=f"{WORK}/subs/queue_submission.csv")
    ap.add_argument("--into", help="full submission CSV whose queue rows are replaced (written as <name>_t2.csv)")
    ap.add_argument("--keep-cache", action="store_true", help="reuse the scored splits' cache (data unchanged)")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    if not SCORED:
        raise SystemExit(f"no scored split with window_index.csv under {RELEASE}/task2/<panel>/")
    spec = {s: v for s, v in spec.items() if s in SCORED}       # a new month may come as one split only
    tags = check_prerequisites(spec)
    print("Task 2 models:", tags, "splits:", list(SCORED), flush=True)
    rebuild_inputs(a.keep_cache)
    from make_final import queue_from_spec
    queue = queue_from_spec(spec, True)
    k = queue_rows(queue)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    k[["window_id", "timestamp", "link_id", "queue_pred"]].to_csv(a.out, index=False)
    print(f"{a.out}: {len(k)} queue rows, {int(k.queue_pred.sum())} predicted queued", flush=True)
    if a.into:
        sub = pd.read_csv(a.into)
        m = dict(zip(k.submission_id.astype(np.int64), k.queue_pred))
        q = sub.task == "queue"
        sub.loc[q, "queue_pred"] = sub.loc[q, "submission_id"].map(m).astype(int)
        dst = a.into[:-4] + "_t2.csv"
        sub.to_csv(dst, index=False)
        print(f"{dst}: queue rows replaced ({int(q.sum())})", flush=True)


if __name__ == "__main__":
    main()
