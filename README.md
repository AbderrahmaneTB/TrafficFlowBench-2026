# TrafficFlowBench 2026 — Team Abderahmane_tb

Solution to the Kaggle competition
[2026 IEEE Big Data — Traffic Flow Bench](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench):
reconstructing, forecasting and explaining freeway traffic on simulated Los Angeles / Orange County corridors.

The full method, experiments and negative results are in the technical report:
**[report/TECHNICAL_REPORT.pdf](report/TECHNICAL_REPORT.pdf)** (13 pages).

## The four tasks

Score: `S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME`.
Public leaderboard = March, private leaderboard = April. Both months run on new traffic scenarios that the nine
training months never show.

| Task | Goal | Our method |
|---|---|---|
| 1. State (0.35) | Fill hidden speed and flow cells of every corridor | Weighted mean of 12 gradient-boosting models (CatBoost, LightGBM, XGBoost) over a time-interpolation base, with month-own profiles and low-rank month completions. A dedicated model handles the 90-minute corridor-wide blanks after each queue-forecast origin. |
| 3. Physics (0.15) | Scored on the Task 1 answer (fundamental diagram, vehicle conservation) | Light time smoothing inside runs of hidden cells, because the conservation score charges the error's variation along time |
| 2. Queue (0.30) | Forecast which links are queued 5–30 min after an origin T | Shift-robust LightGBM (3 seeds) with corridor-state features and the documented window-selection rule. Onset windows: links with probability ≥ 0.5 × the window maximum. Ongoing windows: threshold 0.55 (0.65 for links already queued at T), plus a remote-queue head for second queues far from the current one. |
| 4. ODME (0.20) | Estimate origin–destination path flows | Converged non-negative ridge (λ = 0.05) on each month's own counts and prior |

## Results

| Submission | Public (March) | Private (April) |
|---|---|---|
| Best public probe | 0.89211 | — |
| Selected finals A6 / B6 | 0.88886 | 0.88243 |

The finals keep March settings fixed from training-month evidence only; their April part carries every later
improvement (the organizers allow March evidence for April windows).

**Causality.** Every queue forecast reads only data stamped at or before its own origin T. Ramp files are not used
in Task 2: they are not blanked over the forecast horizon, and on-ramps empty in the same five-minute step as the
mainline jams, so reading them after T would reveal the answer. The organizers rerun Task 2 with all later rows
deleted; `solution/t2_causal_test.py` checks that this leaves every window's inputs unchanged.

## Key findings

* **New scenario per scored month.** Per-link queue shares correlate 0.98–1.00 between training months but only
  0.54–1.00 with March or April, so models are built to transfer: location priors are shifted during training and
  month-own features replace training-month memory.
* **Blank cells.** After each official queue-forecast origin the scored months hide the whole corridor for 90
  minutes. A model trained on emulated blanks was the largest single Task 1 gain (+0.0066 S_total).
* **Remote queues.** The ongoing-queue model only scored links near the current queue, so a second queue forming
  elsewhere could never be predicted. A dedicated head for those links added +0.0022 S_total.
* **Information limit.** The largest remaining queue error is queues that clear before T+30, which depends on
  traffic arriving after T. No decoding of the probabilities fixes it.

## Repository layout

```
solution/            all code (see solution/README.md for every stage)
  run_all.py         single entry point: rebuild the final submission from the official release
  run_task2.py       Task 2 only, for reruns on truncated data or a new month
  make_final.py      assemble a submission from a spec (per-split Task 2 settings)
  specs/             final_a.json / final_b.json — the two selected finals
  t1_*.py            Task 1 (state) models and blank-cell models
  t2_*.py            Task 2 (queue) data, models, remote-queue head, causality test
  t4_*.py            Task 4 (ODME)
report/              technical report (PDF + Markdown source, figures, build scripts)
make_package.py      builds the award package zip (code + PDF)
```

## Reproducing

1. Python 3.12; install `solution/requirements.txt`.
2. Download the competition data (needs a Kaggle account and accepting the rules):
   `kaggle competitions download -c 2026-ieee-big-data-traffic-flow-bench`, then extract it.
3. Set two environment variables and run from `solution/`:

```
export TFB_DATA=<extracted release>      # folder with submission_key.csv, corridors/, task1/ ...
export TFB_WORK=<empty work folder>      # ~35 GB
M="cb6 cb7 cb8 cb9 cb10 cb11:6 cb12:6 cb13:6 lg1 lg2:6 xg1 xg2:6"
python run_all.py --spec specs/final_a.json --name final_a --t1 $M --run-smooth 0.15 0.15
python run_all.py --spec specs/final_b.json --name final_b --t1 $M --run-smooth 0.15 0.15
```

Task 1 members train on a GPU (CatBoost / XGBoost; 1.5–4 h each); everything else runs on CPU. Seeds are fixed;
`run_task2.py` reproduces all 174 000 queue rows of both selected submissions exactly. Details, timings and the
Task 2-only rerun are in [solution/README.md](solution/README.md).

The competition data is not included in this repository.

## Author

Tayeb Boudjema Abderrahmane ([AbderrahmaneTB](https://github.com/AbderrahmaneTB)) — Kaggle team Abderahmane_tb.
