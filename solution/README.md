# TrafficFlowBench 2026 — team Abderahmane_tb — solution code

Rebuilds the final submission (all four tasks) from the official competition release.

## Environment

* Python 3.12; packages: `numpy pandas pyarrow scipy lightgbm catboost xgboost` (versions in `requirements.txt`).
* Task 1 models are trained on a GPU (we used Kaggle notebooks, NVIDIA T4): CatBoost and XGBoost members need one
  GPU; the LightGBM members run on CPU. Everything else runs on CPU (8+ cores, 16 GB RAM).
* Paths are set by two environment variables:
  * `TFB_DATA` — the extracted competition release (the folder that holds `submission_key.csv`, `corridors/`,
    `task1/`, `task2/`, `task4/`, `config/`);
  * `TFB_WORK` — an empty working folder (needs ~35 GB).

## One command

```
M="cb6 cb7 cb8 cb9 cb10 cb11:6 cb12:6 cb13:6 lg1 lg2:6 xg1 xg2:6"
TFB_DATA=<release> TFB_WORK=<work> python run_all.py --spec specs/final_a.json --name final_a --t1 $M --run-smooth 0.15 0.15
TFB_DATA=<release> TFB_WORK=<work> python run_all.py --spec specs/final_b.json --name final_b --t1 $M --run-smooth 0.15 0.15
```

`tag:w` gives a Task 1 member the weight w (default 1): the five feature-v7 members weigh 6, the seven older ones 1.
The two final submissions (Kaggle `fe12w6_bv4568w7s_r15_A6` = final_a and `fe12w6_bv4568w7s_r15_B6` = final_b)
differ only in the Task 2 models used for the April (private) windows (`specs/`): final_a uses v14 seeds 0, 1, 2,
final_b the same recipe with seeds 5, 6, 7. Both use for April ongoing windows the threshold 0.55 (0.65 for links queued at the origin: spec key
`thr_on_q`) and the remote-queue head (`"remote"` in the spec: `t2_remote_head.py` seeds 0, 1, 2, a link outside the ongoing model's
candidate set is queued at a step when the heads' mean probability is ≥ 0.5); their March windows keep the settings
chosen on train-split evidence before any March score (threshold 0.5, no remote head). The second command reuses
every stage of the first, trains the three v14 seeds of final_b (CPU, ~45 min) and re-assembles. Why a seed replica:
Kaggle keeps the better of the two finals on the private board, so final_b should maximise the expected gain
E[(B − A)+]; a replica has the same expected score as A and still differs in about 30 % of the windows. Seeds 5, 6, 7
had the largest value among the 20 three-seed sets of seeds 3–8 on the February and emulated-March windows (train /
March evidence only, as the rules require for April). A reduced run (e.g. `--t1 cb11`) needs one GPU job instead of twelve; from the leaderboard steps in the
report, the full ensemble adds an estimated 0.1–0.2 % over its best single member.

Stages (each one is skipped when its outputs exist, so an interrupted run resumes):

| stage | script(s) | device, time |
|---|---|---|
| 1. dense arrays of every split | `build_cache.py` | CPU, ~5 min |
| 2. Task 1 members, then their mean | `t1_train_cb.py` / `t1_train_xgb.py` / `t1_train_lgb.py`, `t1_predict_cb.py`, `t1_avg_preds.py` | GPU 1.5–4 h per member (LightGBM: CPU) |
| 3. Task 2 training windows | `t2_build.py`, `t2_build3.py`, `t2_month.py`, `t2_sel.py`, `t2_month_trunc.py` | CPU, ~1 h |
| 4. Task 2 models (3 seeds), then the remote-queue heads named in the spec (rh0–rh2) | `t2_train.py`, `t2_remote_head.py` | CPU, ~1.5 h + ~30 min |
| 5. Task 4 | `t4_run.py nnls` | CPU, minutes |
| 6. Task 1 cells in the queue-window blanks: four models (`BLANK_MODELS`; v8 on twice the train-month rows), weighted mean .1 / .1 / .1 / .7 (`BLANK_WEIGHTS`), speed smoothed ±1 slot | `t1_blank_build.py`, `t1_blank_final.py`, `t1_avg_preds.py`, `t1_blank_smooth.py`, `t1_blank_mix.py` | CPU, ~7 h, peak RAM ≈ 7.5 GB (v8) |
| 7. (optional, `--run-smooth AV AQ`) light time smoothing inside runs of target cells | `t1_run_smooth.py` | CPU, ~5 min |
| 8. assembly | `make_final.py` | CPU, ~10 min |

The Task 1 members (feature version, learner, settings) are listed in `T1_MEMBERS` at the top of `run_all.py`; the
final spec file holds the Task 2 settings per scored split.

## Task 2 only (award-review reruns)

```
TFB_DATA=<release> TFB_WORK=<work of the run above> python run_task2.py --spec specs/final_a.json     [--out queue_submission.csv] [--into <full submission>.csv]
```

Rebuilds every input of the scored windows from the data under `TFB_DATA` (cache of the scored splits, the stitched
queued calendar, official-window features t2_build -> t2_build3 -> t2_month -> t2_sel), forecasts with the spec's
models and decision rules, and writes the queue rows of `submission_key.csv` in the official per-task format
(`window_id, timestamp, link_id, queue_pred`, for `merge_submissions.py`); `--into` also writes a copy of a full
submission with its queue rows replaced. Everything it learns from (training windows, location priors, models,
remote-queue heads) is fitted on the train split and taken from `TFB_WORK`. It therefore runs unchanged on the
release with every row after each window's origin removed, and on a newly generated month in the same layout (place
it under `validation/` or `private/`; a month given as one split only is scored with that split's settings — use
`private/` for the April settings of the finals). CPU, about 2 minutes. Checked (Nov 7, finals A6 and B6): from the raw release it reproduces the 174 000 queue rows of the
selected submissions exactly, and `t2_causal_test.py` shows that deleting every row stamped after a window's origin
(masked layers, window history, later window-index rows) leaves that window's inputs bit-identical (160 / 160).

## Seeds, external data, licences

* Seeds are fixed in the code: Task 1 members (`T1_MEMBERS`: CatBoost `--seed` / `--row-seed`, defaults 0), Task 2
  (`SEEDS`: tag suffix → LightGBM / sampling seed, e.g. v14 = 0, v14s1 = 1 … v14s8 = 8; only the tags a spec names are
  trained; v14 training is deterministic: seed 3 retrained on Nov 5 was byte-identical to its Oct 28 model),
  remote-queue heads (`t2_remote_head.py <seed>`: tags rh0–rh2, LightGBM seed 11 + seed), blank
  models (`BLANK_MODELS` `--seed`), row sampling
  in the scripts (`numpy.random.default_rng(seed)`). GPU CatBoost / XGBoost training is deterministic up to float
  summation order; the selected submissions were made from the members trained in the Kaggle notebooks.
* External data: none. No pretrained models, no paid or rate-limited APIs, no proprietary dependencies. Only the
  official competition release is read.
* Software (all open source): numpy, pandas, scipy (BSD-3-Clause), pyarrow (Apache-2.0), LightGBM (MIT),
  CatBoost (Apache-2.0), XGBoost (Apache-2.0); versions in `requirements.txt`.

## What each part does

* **Task 1** (`t1_features.py`, `t1_lowrank.py`, `t1_pseudo.py`): residual models over a time-interpolation base.
  Features: time / space neighbours, month-own profiles, the month's own free-flow level per link, released on-ramp
  flows at the same moment, and low-rank completions of each month (ranks 6–48, alternating ridge least squares on
  the observed cells only). Training rows: train-split targets plus self-supervised rows of the scored months
  (15 % of their observed cells hidden and rebuilt from the doubly-masked layer). Task 1 is offline.
* **Task 1, blank cells** (`t1_blank.py`): in March / April every link of a queue corridor is hidden for the 18 slots
  after each official Task 2 origin, and those cells are Task 1 targets that no ordinary training row resembles.
  Such blanks are emulated at the origins the window selector picks in the scored months (the observed cells are
  the truth), a LightGBM model is fitted on them (features: the Task 1 features of the blanked layer, the blank's
  edges on the cell's and the neighbouring links, on-ramp flows inside the blank, the corridor's demand before and
  after it along the month profile, position in the blank, window condition) and its predictions replace the main
  models' inside the official blanks (`t1_blank_mix.py`). Cross-validation by day parity: `t1_blank_eval.py`.
* **Task 3** is scored on the Task 1 answer. Its conservation term charges the total variation of the Task 1 error
  along time, so the blank speeds are smoothed ±1 slot (`t1_blank_smooth.py`) and the predictions inside runs of
  consecutive target cells are pulled 15 % towards their ±1-slot run mean (`t1_run_smooth.py`).
* **Task 2** (`t2_*.py`): LightGBM on the window history, corridor state features and the documented selection
  procedure; prior-shift augmentation for new scenarios; decision rules in `make_final.py`. **Causality:** every
  feature of a window uses only rows stamped at or before its origin T (the release's 60-minute history, the row at T
  and earlier released rows); ramp files are not used in Task 2. All models are fitted on the train split.
  The ongoing model scores only links near the current queue (`t2_train.candidates`: within 30 links of a queued
  link at T, or queued in the history); a second queue that forms farther away is the job of the remote-queue head
  (`t2_remote_head.py`: same inputs and recipe, fitted only on those remote link-steps of the training windows, the
  step as an input) applied by `t2_remote.py` when a spec has the `"remote"` key.
* **Task 4** (`t4_odme.py`, `t4_run.py`): non-negative ridge (λ = 0.05) on each split's own counts and weak prior.

## Files

`run_task2.py` (Task 2 only, above), `t2_causal_test.py` (deletion test: every official window rebuilt from data
truncated at its origin, compared with the stored inputs), `t2_blankfill.py` (tested, not used: the blank-fill
experiment of the report), `common.py` (paths, network tables), `submission.py` (official upload format), `submit.py` + `kaggle_upload.py`
(chunked upload), `swap_sub.py` (single-change probes), evaluation scripts (`t1_month_eval.py`, `t2_offeval.py`,
`t2_month_eval.py`, ...) used for the experiments in the report.
