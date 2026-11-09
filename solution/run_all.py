"""Single entry script: rebuild the final submission from the official competition data.

  TFB_DATA=<extracted release> TFB_WORK=<empty work dir> python run_all.py --spec specs/final.json --name final
      [--t1 cb6 cb7 cb8]

Stages (each skipped when its outputs already exist, so an interrupted run resumes):
  1. cache        dense numpy arrays of every split                                 (CPU, ~5 min)
  2. task1        each Task 1 member of --t1 (table T1_MEMBERS: feature version, learner, settings), trained on
                  train-split targets + self-supervised March/April rows, then the mean of their predictions
                                                                                    (GPU, ~1.5-4 h per member)
  3. task2 data   selector-emulated windows: data 3 -> 6 -> 9 -> 11                   (CPU, ~1 h)
  4. task2 models LightGBM v13 / v14 (3 seeds each): Feb-held-out run, then full run (CPU, ~1.5 h)
  5. task4        converged non-negative ridge (lambda 0.05) on each split's own data (CPU, minutes)
  6. task1 blanks the cells hidden on every link after each official Task 2 origin get the mean of three models
                  trained on emulated blanks of the scored and training months (t1_blank*.py)   (CPU, ~5 h)
  7. smoothing    optional (--run-smooth): partial time smoothing inside runs of Task 1 target cells (physics)
  8. assemble     make_final.py with per-split Task 2 settings                        (CPU, ~10 min)

Task 2 uses, for every window, only data stamped at or before its origin: models are fitted on the train split,
features of a window come from its released history, the row at T and earlier released rows.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from common import WORK, QUEUE_PANELS, PANELS

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
V11 = ["--aug", "--data", "11", "--drop", "link_pos", "panel", "--prior-shift", "0.5"]
# v13 / v14: v11 + corridor vehicle-count and neighbour-density features (levels 1 / 2)
T2_MODELS = {"v13": V11 + ["--corr", "1"], "v14": V11 + ["--corr", "2"]}
SEEDS = {"": 0, "s1": 1, "s2": 2, "s3": 3, "s4": 4, "s5": 5, "s6": 6, "s7": 7, "s8": 8}   # only tags of the spec train
# Task 1 members: (feature version TFB_T1FEAT, training script, settings). All use the same training rows: 1.2 M
# train-split targets per panel (February held out for early stopping) + every pseudo-target row of March / April.
T1_ROWS = ["--n-train", "1200000", "--n-hold", "150000", "--pseudo", "validation", "private", "--pseudo-nmax", "400000",
           "--pseudo-weight", "2"]
T1_MEMBERS = {
    "cb3": ("2", "t1_train_cb.py", ["--iters", "12000", "--depth", "8"]),
    "cb5": ("3", "t1_train_cb.py", ["--iters", "12000", "--depth", "8"]),
    "cb6": ("4", "t1_train_cb.py", ["--iters", "12000", "--depth", "8"]),
    "cb7": ("5", "t1_train_cb.py", ["--iters", "12000", "--depth", "8"]),
    "cb8": ("6", "t1_train_cb.py", ["--iters", "12000", "--depth", "8"]),
    "cb9": ("6", "t1_train_cb.py", ["--iters", "12000", "--depth", "8", "--pseudo-seeds", "0", "1", "2"]),
    "cb10": ("6", "t1_train_cb.py", ["--iters", "24000", "--depth", "8"]),
    "cb11": ("7", "t1_train_cb.py", ["--iters", "24000", "--depth", "8"]),
    "cb12": ("7", "t1_train_cb.py", ["--iters", "24000", "--depth", "8", "--seed", "7", "--row-seed", "1"]),
    "cb13": ("7", "t1_train_cb.py", ["--iters", "24000", "--depth", "8", "--seed", "13", "--row-seed", "2"]),
    "xg1": ("6", "t1_train_xgb.py", ["--rounds", "10000", "--depth", "9", "--lr", "0.04"]),
    "xg2": ("7", "t1_train_xgb.py", ["--rounds", "16000", "--depth", "9", "--lr", "0.04", "--seed", "11"]),
    "lg1": ("6", "t1_train_lgb.py", ["--rounds", "6000", "--leaves", "255", "--lr", "0.05"]),
    "lg2": ("7", "t1_train_lgb.py", ["--rounds", "6000", "--leaves", "255", "--lr", "0.05"]),
    "lg3": ("7", "t1_train_lgb.py", ["--rounds", "6000", "--leaves", "255", "--lr", "0.05", "--seed", "17", "--row-seed", "3"]),
    "lg4": ("7", "t1_train_lgb.py", ["--rounds", "6000", "--leaves", "255", "--lr", "0.05", "--seed", "23", "--row-seed", "4"]),
}


# blank models (t1_blank_final.py), averaged and then smoothed along time (speed): settings chosen by the day-parity
# cross-validation (t1_blank_eval.py); the seeds and data shares differ so that the mean reduces variance
BLANK_COMMON = ["--splits", "train", "validation", "private", "--groups", "0", "1", "2", "--extra", "--flow-base", "simple",
                "--leaves", "255", "--min-leaf", "50", "--flow-mix", "0.7"]
BLANK_MODELS = {
    "v4": ["--train-frac", "0.5", "--rounds-v", "4500", "--rounds-q", "4500", "--seed", "0"],
    "v5": ["--train-frac", "1.0", "--rounds-v", "5000", "--rounds-q", "5000", "--seed", "1"],
    "v6": ["--train-frac", "1.0", "--rounds-v", "5000", "--rounds-q", "4000", "--seed", "2", "--extra2"],
    # twice the train-month blank rows (train groups 0 + 1): -3 % speed RMSE in the cross-validation
    "v8": ["--train-groups", "0", "1", "--train-frac", "1.0", "--rounds-v", "6000", "--rounds-q", "5000", "--seed", "4",
           "--extra2"],
}
# weights of the blank mean: the 2x-rows model carries .7 (out-of-fold mix test), the three smaller ones .1 each
BLANK_WEIGHTS = {"v4": 1, "v5": 1, "v6": 1, "v8": 7}


def run(*args, env=None):
    t0 = time.time()
    print(">>>", " ".join(args), flush=True)
    subprocess.run([PY, "-u", "-W", "ignore", *args], cwd=HERE, check=True, env=env)
    print(f"<<< {time.time() - t0:.0f}s", flush=True)


def exists(*paths):
    return all(os.path.exists(f"{WORK}/{p}") for p in paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--name", default="final")
    ap.add_argument("--task-type", default="GPU", help="CatBoost / XGBoost device for Task 1 training")
    ap.add_argument("--t1", nargs="+", default=["cb6", "cb7", "cb8"],
                    help="Task 1 members (T1_MEMBERS), averaged; 'tag:w' gives a member weight w (default 1)")
    ap.add_argument("--run-smooth", nargs=2, type=float, metavar=("AV", "AQ"),
                    help="partial time smoothing of speed / flow inside runs of target cells (t1_run_smooth.py)")
    a = ap.parse_args()
    if not exists(*[f"cache/{p}_{s}.npz" for p in PANELS for s in ("train", "validation", "private")]):
        run("build_cache.py")
    gpu = a.task_type == "GPU"
    weights = {t.split(":")[0]: float(t.split(":")[1]) if ":" in t else 1.0 for t in a.t1}
    for tag in weights:
        feat, script, extra = T1_MEMBERS[tag]
        env1 = dict(os.environ, TFB_T1FEAT=feat, TFB_XGB_DEVICE="cuda" if gpu else "cpu")
        ext = {"t1_train_xgb.py": "ubj", "t1_train_lgb.py": "lgb"}.get(script, "cbm")
        if exists(*[f"t1pred_{tag}/{p}_{s}.npz" for p in PANELS for s in ("validation", "private")]):
            continue                               # predictions present (e.g. made on a GPU machine)
        if not exists(f"t1_speed_{tag}.{ext}", f"t1_flow_{tag}.{ext}"):
            dev = {"ubj": ["--device", "cuda" if gpu else "cpu"], "lgb": []}.get(ext, ["--task-type", a.task_type])
            run(script, "--tag", tag, *T1_ROWS, *extra, *dev, env=env1)
        run("t1_predict_cb.py", "--tags", tag, "--out", f"t1pred_{tag}", "--task-type", "CPU", env=env1)
    run("t1_avg_preds.py", "--out", "t1pred_final", *[f"t1pred_{tag}:{w}" for tag, w in weights.items()])
    # t2_build.py first: it also writes the official validation/private windows that t2_build3.py extends
    for script, data in (("t2_build.py", ""), ("t2_build3.py", "3"), ("t2_month.py", "6"), ("t2_sel.py", "9"),
                         ("t2_month_trunc.py", "11")):
        if not exists(*[f"t2/{s}{data}_{p}.npz" for p in QUEUE_PANELS for s in ("train", "validation", "private")]):
            run(script)
    spec = json.load(open(a.spec))
    needed = {t for s in spec.values() for c in ("onset", "ongoing") for t in s[c]}
    for base, args in T2_MODELS.items():
        for sfx, seed in SEEDS.items():
            tag = base + sfx                       # v13, v13s1, v13s2, v14, v14s1, v14s2
            if tag not in needed:
                continue
            if not exists(f"t2/ongoing_{tag}.txt", f"t2/rounds_{tag}.json"):
                run("t2_train.py", "--tag", tag, "--seed", str(seed), *args)
            if not exists(f"t2/ongoing_{tag}_full.txt"):
                run("t2_train.py", "--tag", tag, "--seed", str(seed), *args, "--full")
    # optional remote-queue heads of ongoing windows (t2_remote.py), tags rh<seed>: Feb-held-out run, then full
    remote = {t for s in spec.values() if s.get("remote", {}).get("kind") == "head" for t in s["remote"]["tags"]}
    for tag in sorted(remote):
        seed = tag[2:]
        if not exists(f"t2/remote_{tag}.txt", f"t2/rounds_{tag}.json"):
            run("t2_remote_head.py", seed)
        if not exists(f"t2/remote_{tag}_full.txt"):
            run("t2_remote_head.py", seed, "--full")
    if not exists("t4/nnls.npz"):
        run("t4_run.py", "nnls")
    # Task 1 cells inside the panel-wide blanks after each official Task 2 origin: dedicated model (needs the official
    # window files written by t2_build.py above)
    env7 = dict(os.environ, TFB_T1FEAT="7")
    run("t1_blank_build.py", "--splits", "validation", "private", "--groups", "0", "1", "2", "--ngroups", "3", env=env7)
    run("t1_blank_build.py", "--splits", "train", "--groups", "0", "1", "--ngroups", "4", "--extras", env=env7)
    for tag, extra in BLANK_MODELS.items():
        if not exists(*[f"t1pred_blank_{tag}/{p}_{s}.npz" for p in PANELS for s in ("validation", "private")]):
            run("t1_blank_final.py", *BLANK_COMMON, *extra, "--out", f"t1pred_blank_{tag}", env=env7)
    run("t1_avg_preds.py", "--out", "t1pred_blank", *[f"t1pred_blank_{tag}:{BLANK_WEIGHTS[tag]}" for tag in BLANK_MODELS])
    run("t1_blank_smooth.py", "--inp", "t1pred_blank", "--out", "t1pred_blank_s", "--w", "1")
    run("t1_blank_mix.py", "--base", "t1pred_final", "--blank", "t1pred_blank_s", "--out", "t1pred_final_b")
    t1 = "t1pred_final_b"
    if a.run_smooth:
        run("t1_run_smooth.py", "--inp", t1, "--out", "t1pred_final_r", "--av", str(a.run_smooth[0]),
            "--aq", str(a.run_smooth[1]))
        t1 = "t1pred_final_r"
    run("make_final.py", "--name", a.name, "--t1", t1, "--spec", a.spec)


if __name__ == "__main__":
    main()
