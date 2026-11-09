"""Figures for the technical report (static PNG for the PDF). Run from solution/'s parent:
    python report/make_figures.py
Writes report/fig/*.png. Numbers in figs 2-4 are the measured values quoted in REPORT_DRAFT.md."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "solution"))
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (fixed order)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def fig_freeflow():
    """Per-link free-flow speed level: February (train scenario) vs March and April (new scenarios)."""
    from common import PANELS, load, free_speed
    from t1_features import month_free_speed
    xs, ym, ya = [], [], []
    for p in PANELS:
        vfh = free_speed(p)
        mvf_t, _ = month_free_speed(load(p, "train"), vfh)          # one row per train month, last = 2031-02
        mvf_m, _ = month_free_speed(load(p, "validation"), vfh)
        mvf_a, _ = month_free_speed(load(p, "private"), vfh)
        xs.append(mvf_t[-1]); ym.append(mvf_m[0]); ya.append(mvf_a[0])
    x, ym, ya = np.concatenate(xs), np.concatenate(ym), np.concatenate(ya)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    lo, hi = 100, 121.5
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(x, ym, s=9, color=S1, alpha=0.75, lw=0, label="March", zorder=2)
    ax.scatter(x, ya, s=9, color=S2, alpha=0.75, lw=0, label="April", zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("February free-flow level per link (km/h)")
    ax.set_ylabel("Same link in the scored month (km/h)")
    ax.set_title("Each scenario redraws the free-flow level of every link", loc="left")
    d1, d2 = np.abs(ym - x), np.abs(ya - x)
    ax.text(0.02, 0.97, f"mean |shift| {np.nanmean(d1):.2f} km/h (March), {np.nanmean(d2):.2f} (April)\n"
                        "speed noise 1.2 km/h; truth capped at 120.75", transform=ax.transAxes, va="top",
            color=INK2, fontsize=8)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(OUT, "fig1_freeflow_shift.png")); plt.close(fig)


def fig_lowrank():
    """Task 1 feature additions, like-for-like harness (LightGBM, 60k train rows/panel + March pseudo rows)."""
    steps = ["month-own\nprofiles", "+ rank-6\nfit", "+ month free-\nflow level", "+ rank-12\nfit", "+ rank-24\nfit",
             "+ rank-48\nfit"]
    march = [0.9290, 0.9339, 0.9343, 0.9358, 0.9369, 0.9379]
    feb = [0.9372, 0.9407, 0.9412, 0.9423, 0.9432, 0.9437]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    xi = np.arange(len(steps))
    ax.plot(xi, feb, color=S2, lw=2, marker="o", ms=5, label="February targets (train scenario)")
    ax.plot(xi, march, color=S1, lw=2, marker="o", ms=5, label="Held-out March days (new scenario)")
    for k in (0, len(steps) - 1):
        ax.annotate(f"{march[k]:.4f}", (xi[k], march[k]), textcoords="offset points", xytext=(6, -12),
                    ha="left", color=INK2, fontsize=8)
        ax.annotate(f"{feb[k]:.4f}", (xi[k], feb[k]), textcoords="offset points", xytext=(0, 7),
                    ha="center", color=INK2, fontsize=8)
    ax.set_xticks(xi); ax.set_xticklabels(steps, fontsize=8)
    ax.set_ylabel("S_state")
    ax.set_ylim(0.927, 0.9455)
    ax.set_title("Task 1: month-level structure closes part of the scenario gap", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig2_task1_features.png")); plt.close(fig)


def fig_onset_rule():
    """Onset decision rules when the location priors are wrong in a share of the windows (train-split test)."""
    shift = [0, 50, 100]
    rules = {"P ≥ 0.5 × window max (final)": ([0.878, 0.849, 0.802], S1),
             "P ≥ 0.5 (fixed threshold)": ([0.887, 0.843, 0.770], S2),
             "P ≥ 0.4 × window max": ([0.875, 0.856, 0.831], S3)}
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for name, (v, c) in rules.items():
        ax.plot(shift, v, color=c, lw=2, marker="o", ms=5, label=name)
    ax.set_xticks(shift); ax.set_xticklabels(["0 %", "50 %", "100 %"])
    ax.set_xlabel("Share of windows with location priors rolled to wrong links")
    ax.set_ylabel("Onset IoU (February, official-like windows)")
    ax.set_title("Onset: hedging over candidate bottlenecks", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig3_onset_rule.png")); plt.close(fig)


def fig_leaderboard():
    """Public leaderboard (March) after each single-component change."""
    steps = [("s1", 0.82136, "baseline"), ("s2", 0.85160, "ODME\nNNLS"),
             ("s13", 0.85675, "state\nCatBoost"), ("s14", 0.86182, "queue\nrobust"),
             ("s38", 0.86852, "queue\ncorridor"), ("F5", 0.87998, "queue ×3,\nonset rule"),
             ("F8", 0.88058, "state\nmonth fits"), ("fe6to10", 0.88120, "state\n5 models"),
             ("p2", 0.88535, "blanks:\ninterp."), ("p3", 0.88711, "blanks:\nmodel"),
             ("Nov 1", 0.88852, "blank\nbag"), ("Nov 2", 0.88878, "12\nmodels"),
             ("final", 0.88886, "v7 ×6,\n4 blanks")]
    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    xi = np.arange(len(steps)); ys = [s[1] for s in steps]
    ax.plot(xi, ys, color=S1, lw=2, marker="o", ms=5)
    for k in (0, len(steps) - 1):
        ax.annotate(f"{ys[k]:.5f}", (k, ys[k]), textcoords="offset points", xytext=(8 if k == 0 else 0, 8),
                    ha="left" if k == 0 else "center",
                    fontsize=8, color=INK2)
    ax.set_xticks(xi); ax.set_xticklabels([f"{s[0]}\n{s[2]}" for s in steps], fontsize=7)
    ax.set_ylabel("Public S_total")
    ax.set_ylim(0.815, 0.895)
    ax.set_title("Public leaderboard (March), main steps", loc="left")
    fig.savefig(os.path.join(OUT, "fig4_leaderboard.png")); plt.close(fig)


def fig_t2_errors():
    """Ongoing IoU by what the true queue does over the horizon (solution/t2_error_analysis.py output). Dot plot: the
    axis does not start at zero, so the values are positions, not bar lengths."""
    import json
    from common import WORK
    r = json.load(open(f"{WORK}/t2_error_analysis.json"))
    kinds = ["about stable", "shrinks", "grows", "dissolves"]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    yi = np.arange(len(kinds))[::-1]
    for key, lab, c, off in (("february", "February (train scenario)", S1, 0.12),
                             ("emulated_march", "Emulated March (new scenario)", S2, -0.12)):
        vals = [r[key][f"ongoing_{k}"][1] for k in kinds]
        ns = [r[key][f"ongoing_{k}"][0] for k in kinds]
        ax.scatter(vals, yi + off, s=42, color=c, label=lab, zorder=3, edgecolor=SURF, linewidth=1.5)
        for v, y, n in zip(vals, yi + off, ns):
            ax.annotate(f"{v:.2f} (n={n})", (v, y), textcoords="offset points", xytext=(7, -3), fontsize=7, color=INK2)
    ax.set_yticks(yi); ax.set_yticklabels([f"queue {k}" for k in kinds], fontsize=8)
    ax.set_xlim(0.70, 1.0)
    ax.set_xlabel("Ongoing-window IoU")
    ax.set_title("Ongoing windows: dissolving queues are the hard case", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(OUT, "fig5_t2_ongoing_errors.png")); plt.close(fig)


def fig_blanks():
    """Task 1 cells inside the queue-window blanks: error of each predictor on emulated blanks (cross-validated by day
    parity, the same 1.37 million held-out cells in every row; t1_blank_eval.py). Dot plot, one panel per variable."""
    rows = [("time interpolation\n(flow: + corridor demand)", (9.69, 10.26), (98.7, 97.6)),
            ("blank model,\nMarch / April blanks", (6.32, 6.70), (83.0, 80.8)),
            ("+ larger trees", (6.24, 6.64), (82.9, 80.8)),
            ("+ training-month\nblanks", (5.72, 6.29), (80.2, 79.1))]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    yi = np.arange(len(rows))[::-1]
    for ax, k, unit in ((axes[0], 1, "speed RMSE, km/h"), (axes[1], 2, "flow RMSE, veh/h/lane")):
        for j, (lab, c, off) in enumerate((("March", S1, 0.12), ("April", S2, -0.12))):
            vals = [r[k][j] for r in rows]
            ax.scatter(vals, yi + off, s=42, color=c, label=lab, zorder=3, edgecolor=SURF, linewidth=1.5)
            for v, y in zip(vals, yi + off):
                ax.annotate(f"{v:g}", (v, y), textcoords="offset points", xytext=(6, -3), fontsize=7, color=INK2)
        ax.set_xlabel(unit)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(5.0, 11.5); axes[1].set_xlim(75, 104)
    axes[0].set_yticks(yi); axes[0].set_yticklabels([r[0] for r in rows], fontsize=7.5)
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Cells inside the queue-window blanks: error on emulated blanks", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", color=INK)
    fig.savefig(os.path.join(OUT, "fig6_blanks.png")); plt.close(fig)


if __name__ == "__main__":
    fig_lowrank(); fig_onset_rule(); fig_leaderboard(); fig_freeflow(); fig_t2_errors(); fig_blanks()
    print("figures in", OUT)
