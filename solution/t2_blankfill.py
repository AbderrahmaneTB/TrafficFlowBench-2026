"""Task 2: fill the panel-wide blanks of earlier official windows in the queued calendar used by the month features.

The scored layers (March, April) hide every link of a queue panel for the 18 slots after each official origin (the
Task 1 blanks, T+5 ... T+90 min); the nine train months have no such blanks. A later window's month features read the
earlier days of its month (mq_now / mq_h6 / mq_dayany / mq_allday, t2_month.py), and on 34-49 % of those days a blank
covers exactly the block the feature reads - the queue period of that day. The onset model takes 45 % of its gain from
the month features (mq_h6_z alone 19 %), so in the scored months it reads a calendar biased towards "no queue" that it
never saw in training.

The ramp detectors are never blanked, and an on-ramp's flow collapses in the step its mainline link jams. A LightGBM
classifier fitted on the TRAIN months (blocks T+1 ... T+18 after the train windows, label = the observed queued state)
predicts the queued state of every blanked cell from the ramps inside the block and the mainline state at the block's
two edges; the scored calendar gets these 0 / 1 values on eligible blank cells.

Legal for Task 2: a window at origin T reads the calendar of EARLIER days of its month and of its own day before
T - 60 min only (t2_month.month_features). A block of an earlier window on those rows is filled from its own day's
ramps inside the block and the mainline cells at most 3 slots beyond its end (T' + 22), all stamped before T (windows
of one day are >= 360 min apart), with a model fitted on the train months and ramp levels normalised by train-month
means. `deletion_test()` checks this for every official window.

  python t2_blankfill.py fit        -> work/t2/blankfill_{nofeb,all}.txt (+ held-out February report)
  python t2_blankfill.py calendar   -> work/t2/qcal_fill_<panel>.npz (scored blanks filled by the 'all' model)
  python t2_blankfill.py month      -> work/t2/{validation,private}11F_<panel>.npz, validationM11F_<panel>.npz
                                       (data 11 with the month columns recomputed from the filled calendar)
  python t2_blankfill.py test       -> deletion test on every official window
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import WORK, QUEUE_PANELS, load, network
from t2_recent import queued_calendar

OUT = f"{WORK}/t2"
BL = 18
EDGE = 3
NB = (("u2", -2), ("u1", -1), ("d1", 1), ("d2", 2))
NAMES = (["k", "cond", "slot", "sL", "sR", "vL", "vR"]
         + [f"{s}_{n}" for n, _ in NB for s in ("sL", "sR")]
         + ["dqdn_L", "dqup_L", "dqdn_R", "dqup_R", "run_L", "run_R", "nq_L", "nq_R",
            "ron_m1", "ron_0", "ron_p1", "ron_min2", "ron_min5", "has_on", "roff_0", "roff_min2", "low_share",
            "ron_L", "ron_R"])
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=8)


# ----------------------------------------------------------------------------------------------- ramp signal
def ramp_norm(panel: str) -> dict:
    """Mean flow per ramp and slot over the train months (valid cells), floor 20 veh/h: {ramp id: (288,)}."""
    zt = load(panel, "train")
    rq = zt["rq"]
    ok = (zt["relig"] > 0) & np.isfinite(rq)
    with np.errstate(all="ignore"):
        m = np.where(ok, rq, 0).sum(0) / ok.sum(0)
    m = np.maximum(np.nan_to_num(m, nan=20.0), 20.0)
    return {str(r): m[:, k].astype(np.float32) for k, r in enumerate(zt["ramps"])}


def ramp_ratio(z: dict, norm: dict, kinds) -> np.ndarray:
    """(D, 288, L) smallest flow ratio (flow / train-month slot mean) over the ramps of the given types attached to
    each link; NaN where the link has no observed ramp of that type."""
    links = {str(x): i for i, x in enumerate(z["links"])}
    D, T, L = z["mv"].shape
    out = np.full((D, T, L), np.nan, np.float32)
    for r, (rid, typ, lk) in enumerate(zip(z["ramps"], z["ramp_type"], z["ramp_link"])):
        j = links.get(str(lk))
        if j is None or str(typ).upper() not in kinds or str(rid) not in norm:
            continue
        ok = (z["relig"][..., r] > 0) & np.isfinite(z["rq"][..., r])
        ratio = np.where(ok, z["rq"][..., r] / norm[str(rid)][None, :], np.nan)
        a = out[:, :, j]
        out[:, :, j] = np.where(np.isnan(a), ratio, np.where(np.isnan(ratio), a, np.minimum(a, ratio)))
    return out


# ----------------------------------------------------------------------------------------------- block features
def _nb(a: np.ndarray, off: int) -> np.ndarray:
    """Value at link j + off along the last axis (NaN outside the corridor)."""
    out = np.full(a.shape, np.nan, np.float32)
    if off > 0:
        out[..., :-off] = a[..., off:]
    elif off < 0:
        out[..., -off:] = a[..., :off]
    else:
        out[...] = a
    return out


def _edge(Qd, Rd, slots):
    L = Qd.shape[1]
    st = np.full(L, np.nan, np.float32); vr = np.full(L, np.nan, np.float32)
    for s in slots:
        if 0 <= s < Qd.shape[0]:
            ok = np.isnan(st) & (Qd[s] >= 0)
            st[ok] = Qd[s][ok]; vr[ok] = Rd[s][ok]
    return st, vr


def _dist_run(st, sign):
    L = len(st)
    q = np.flatnonzero(st == 1)
    ar = np.arange(L)
    if len(q):
        off = sign * (q[None, :] - ar[:, None])
        dn = np.where(off >= 0, off, 10 ** 6).min(axis=1).astype(np.float32)
        up = np.where(off <= 0, -off, 10 ** 6).min(axis=1).astype(np.float32)
        dn[dn >= 10 ** 6] = 99; up[up >= 10 ** 6] = 99
    else:
        dn = np.full(L, 99, np.float32); up = np.full(L, 99, np.float32)
    run = np.zeros(L, np.float32)
    s0 = 0
    while s0 < L:
        if st[s0] == 1:
            e0 = s0
            while e0 < L and st[e0] == 1:
                e0 += 1
            run[s0:e0] = e0 - s0; s0 = e0
        else:
            s0 += 1
    return dn, up, run


def block_features(Qd, Rd, ron, roff, T: int, cond: int, sign: int):
    """Features (n_cells, F) of the cells T+1 ... T+BL (slot-major, all links) of one day.
    Qd (288, L) int8 observed queued state (-1 unobserved); Rd (288, L) speed / vcut; ron / roff (288, L) ramp
    ratios. Only Qd / Rd rows <= T and T+BL+1 ... T+BL+1+EDGE are read (the block itself is never looked at)."""
    L = Qd.shape[1]
    slots = np.arange(T + 1, min(T + BL, 287) + 1)
    sL, vL = _edge(Qd, Rd, range(T, T - EDGE - 1, -1))
    sR, vR = _edge(Qd, Rd, range(T + BL + 1, T + BL + 2 + EDGE))
    dL, uL, rL = _dist_run(sL, sign)
    dR, uR, rR = _dist_run(sR, sign)
    per_link = [sL, sR, vL, vR]
    for _, o in NB:
        per_link += [_nb(sL, sign * o), _nb(sR, sign * o)]
    per_link += [dL, uL, dR, uR, rL, rR, np.full(L, np.nansum(sL == 1), np.float32),
                 np.full(L, np.nansum(sR == 1), np.float32)]
    has_on = np.isfinite(ron).any(axis=0).astype(np.float32)
    ron_L = ron[T] if T < 288 else np.full(L, np.nan)
    ron_R = ron[min(T + BL + 1, 287)]
    rows = []
    with np.errstate(all="ignore"):
        for t in slots:
            r0 = ron[t]; rm = ron[max(t - 1, 0)]; rp = ron[min(t + 1, 287)]
            m2 = np.nanmin(np.stack([_nb(r0, o) for o in range(-2, 3)]), axis=0)
            m5 = np.nanmin(np.stack([_nb(r0, o) for o in range(-5, 6)]), axis=0)
            f0 = roff[t]
            fm2 = np.nanmin(np.stack([_nb(f0, o) for o in range(-2, 3)]), axis=0)
            fin = np.isfinite(r0)
            low = float(np.mean(r0[fin] < 0.3)) if fin.any() else np.nan
            rows.append(np.column_stack([np.full(L, t - T), np.full(L, cond), np.full(L, t), *per_link,
                                         rm, r0, rp, m2, m5, has_on, f0, fm2, np.full(L, low), ron_L, ron_R]))
    return np.concatenate(rows).astype(np.float32), slots


# ----------------------------------------------------------------------------------------------- panel layers
class Layer:
    """Queued state, speed ratio and ramp ratios of one split of one panel."""

    def __init__(self, panel: str, split: str, norm: dict):
        z = load(panel, split)
        net = network(panel)
        self.vcut = net.vcut.to_numpy(np.float32)
        self.sign = int(net.attrs["direction"])
        v = z["mv"]; el = z["melig"].astype(bool)
        self.el = el
        self.Q = np.where(np.isfinite(v) & el, (v <= self.vcut).astype(np.int8), np.int8(-1))
        with np.errstate(all="ignore"):
            self.R = (v / self.vcut).astype(np.float32)
        self.ron = ramp_ratio(z, norm, ("OR", "ON"))
        self.roff = ramp_ratio(z, norm, ("FR", "OFF"))
        self.dates = [str(x) for x in z["dates"]]

    def features(self, d: int, T: int, cond: int):
        return block_features(self.Q[d], self.R[d], self.ron[d], self.roff[d], T, cond, self.sign)


def train_blocks(panel: str):
    """(day, slot, cond, month) of the blocks after the train windows (train3), one per window."""
    z = np.load(f"{OUT}/train3_{panel}.npz", allow_pickle=True)
    out = []
    for i, c, dt in zip(z["meta_i"], z["meta_cond"], z["meta_date"]):
        i = int(i)
        out.append((i // 288, i % 288, int(str(c) == "queue_ongoing"), str(dt)[:7]))
    return sorted(set(out))


def official_blocks(panel: str, split: str):
    """(day, slot, cond) of the official windows of a scored split (their panel-wide blanks)."""
    e = np.load(f"{OUT}/{split}3_{panel}.npz", allow_pickle=True)
    dates = [str(x) for x in load(panel, split)["dates"]]
    di = {d: k for k, d in enumerate(dates)}
    out = []
    for o, c in zip(e["origin"], e["cond"]):
        T = pd.Timestamp(str(o))
        out.append((di[T.strftime("%Y-%m-%d")], T.hour * 12 + T.minute // 5, int(str(c) == "queue_ongoing")))
    return out


def collect(frac=0.4, full_months=("2031-02",), seed=0):
    """{month: (X, y)} over the blocks after the train windows of all queue panels; observed cells only, a random
    `frac` of them except in `full_months` (kept whole for the held-out report)."""
    rng = np.random.default_rng(seed)
    acc = {}
    for p in QUEUE_PANELS:
        lay = Layer(p, "train", ramp_norm(p))
        for d, T, c, mon in train_blocks(p):
            if T + 1 > 287:
                continue
            F, slots = lay.features(d, T, c)
            y = lay.Q[d][slots].reshape(-1)
            ok = y >= 0
            if mon not in full_months:
                ok &= rng.random(len(y)) < frac
            a = acc.setdefault(mon, ([], []))
            a[0].append(F[ok]); a[1].append(y[ok])
        print(p, "blocks done", flush=True)
    return {m: (np.concatenate(a[0]), np.concatenate(a[1]).astype(np.float32)) for m, a in acc.items()}


def stack(rows, months, frac=1.0, seed=1):
    X = np.concatenate([rows[m][0] for m in months]); y = np.concatenate([rows[m][1] for m in months])
    if frac < 1.0:
        k = np.random.default_rng(seed).random(len(y)) < frac
        X, y = X[k], y[k]
    return X, y


def report(p, y, F):
    """Cell accuracy and IoU of the queued cells: model vs the edge-persistence baseline."""
    pr = p >= 0.5
    ci = {n: i for i, n in enumerate(NAMES)}
    base = np.where(np.isfinite(F[:, ci["sL"]]), F[:, ci["sL"]], F[:, ci["sR"]]) == 1
    out = {}
    for name, q in (("model", pr), ("edge_persist", base), ("all_free", np.zeros_like(pr))):
        t = y == 1
        u = (q | t).sum()
        out[name] = dict(acc=round(float(np.mean(q == t)), 4), iou=round(float((q & t).sum() / max(u, 1)), 4))
    return out


def fit():
    rows = collect()
    months = sorted(rows)
    print("train months", months, {m: len(rows[m][1]) for m in months}, flush=True)
    noJF = [m for m in months if m not in ("2031-01", "2031-02")]
    Xa, ya = stack(rows, noJF)
    Xe, ye = stack(rows, ["2031-01"])
    print("fit rows", Xa.shape, "queued share", round(float(ya.mean()), 3), "| January early-stop rows", Xe.shape, flush=True)
    da = lgb.Dataset(Xa, ya, feature_name=NAMES, free_raw_data=False)
    de = lgb.Dataset(Xe, ye, reference=da)
    b = lgb.train(PARAMS, da, 3000, valid_sets=[de], callbacks=[lgb.early_stopping(100, verbose=False)])
    n_best = b.best_iteration
    print("best iteration", n_best, flush=True)
    # held-out February blocks: quality of the fill (this model saw neither January as fit rows nor February)
    Xf, yf = stack(rows, ["2031-02"])
    print("February held-out cells", Xf.shape, report(b.predict(Xf, num_iteration=n_best), yf, Xf), flush=True)
    for cond in (0, 1):
        k = Xf[:, 1] == cond
        print("  cond", "ongoing" if cond else "onset", report(b.predict(Xf[k], num_iteration=n_best), yf[k], Xf[k]),
              flush=True)
    # 'nofeb' model for the February evidence: fitted on Jun-Jan; 'all' model for the scored months
    Xb, yb = stack(rows, [m for m in months if m != "2031-02"])
    lgb.train(PARAMS, lgb.Dataset(Xb, yb, feature_name=NAMES), n_best).save_model(f"{OUT}/blankfill_nofeb.txt")
    del Xb, yb
    Xc, yc = stack(rows, months[:-1] + ["2031-02"], frac=1.0)
    kF = np.concatenate([np.full(len(rows[m][1]), m == "2031-02") for m in months[:-1] + ["2031-02"]])
    keep = ~kF | (np.random.default_rng(2).random(len(kF)) < 0.4)    # February subsampled like the other months
    b3 = lgb.train(PARAMS, lgb.Dataset(Xc[keep], yc[keep], feature_name=NAMES), n_best)
    b3.save_model(f"{OUT}/blankfill_all.txt")
    g = b3.feature_importance("gain"); o = np.argsort(-g)
    print("top gain", [(NAMES[i], round(100 * g[i] / g.sum(), 1)) for i in o[:12]], flush=True)


# ----------------------------------------------------------------------------------------------- filling
def fill_blocks(Q: np.ndarray, el: np.ndarray, lay: Layer, blocks, model, day_offset: int = 0) -> np.ndarray:
    """Q (calendar) with the cells of each block (layer day d, slot T, cond) set to the model's 0 / 1 state on
    eligible cells. day_offset maps layer days to calendar rows."""
    Q = Q.copy()
    for d, T, c in blocks:
        F, slots = lay.features(d, T, c)
        p = model.predict(F).reshape(len(slots), -1)
        q = (p >= 0.5).astype(np.int8)
        cur = Q[day_offset + d, slots]
        Q[day_offset + d, slots] = np.where(el[d][slots], q, cur)
    return Q


def calendar():
    m = lgb.Booster(model_file=f"{OUT}/blankfill_all.txt")
    for p in QUEUE_PANELS:
        Q, cal = queued_calendar(p)
        cidx = {d: k for k, d in enumerate(cal)}
        norm = ramp_norm(p)
        changed = 0
        for split in ("validation", "private"):
            lay = Layer(p, split, norm)
            off = cidx[lay.dates[0]]
            assert cal[off:off + len(lay.dates)] == lay.dates
            blocks = official_blocks(p, split)
            before = Q.copy()
            Q = fill_blocks(Q, lay.el, lay, blocks, m, day_offset=off)
            changed += int((before != Q).sum())
        np.savez(f"{OUT}/qcal_fill_{p}.npz", Q=Q, dates=np.array(cal))
        print(p, "filled cells", changed, flush=True)


RECENT3 = ["rq7_now", "rq7_h6", "rq7_dayany"]


def calendar_columns(Q, cal, day: int, slot: int, names: list) -> tuple[list, np.ndarray]:
    """Column indices and (L, k) values of every feature read from the queued calendar: the month features and the
    previous-7-days features rq7_* (t2_recent.recent_features), for a window at calendar row `day`, slot `slot`."""
    from t2_month import MONTH_NAMES, month_features
    from t2_recent import recent_features
    cols = [names.index(n) for n in MONTH_NAMES] + [names.index(n) for n in RECENT3]
    vals = np.column_stack([month_features(Q, cal, day, slot), np.stack(recent_features(Q, day, slot), axis=1)])
    return cols, vals


def month(suffix="11F"):
    """Copies of data 11 (validation / private official windows) and of the emulated March windows (validationM)
    with the calendar columns (month features + rq7_*) recomputed from the filled calendar."""
    for p in QUEUE_PANELS:
        zq = np.load(f"{OUT}/qcal_fill_{p}.npz")
        Q, cal = zq["Q"], [str(x) for x in zq["dates"]]
        cidx = {d: k for k, d in enumerate(cal)}
        for split in ("validation", "private"):
            e = dict(np.load(f"{OUT}/{split}11_{p}.npz", allow_pickle=True))
            names = [str(n) for n in e["names"]]
            X = e["X"].copy()
            for w, o in enumerate(e["origin"]):
                T = pd.Timestamp(str(o))
                cols, vals = calendar_columns(Q, cal, cidx[T.strftime("%Y-%m-%d")], T.hour * 12 + T.minute // 5, names)
                X[w][:, cols] = vals
            dw = np.any(np.nan_to_num(X, nan=-7) != np.nan_to_num(e["X"], nan=-7), axis=(1, 2))
            print(p, split, "windows with changed calendar features", int(dw.sum()), "of", len(dw), flush=True)
            e["X"] = X
            np.savez(f"{OUT}/{split}{suffix}_{p}.npz", **e)
        e = dict(np.load(f"{OUT}/validationM_{p}.npz", allow_pickle=True))
        names = [str(n) for n in e["names"]]
        X = e["X"].copy()
        for w, (i, dt) in enumerate(zip(e["meta_i"], e["meta_date"])):
            cols, vals = calendar_columns(Q, cal, cidx[str(dt)], int(i) % 288, names)
            X[w][:, cols] = vals
        e["X"] = X
        np.savez(f"{OUT}/validationM{suffix}_{p}.npz", **e)


def deletion_test():
    """For every official window: rebuild the filled calendar from the rows stamped at or before its origin only
    (mainline and ramps after T deleted, model and ramp normalisation from the train months) and check that its month
    features equal the stored ones."""
    m = lgb.Booster(model_file=f"{OUT}/blankfill_all.txt")
    same = total = 0
    for p in QUEUE_PANELS:
        Q0, cal = queued_calendar(p)
        cidx = {d: k for k, d in enumerate(cal)}
        norm = ramp_norm(p)
        for split in ("validation", "private"):
            lay = Layer(p, split, norm)
            off = cidx[lay.dates[0]]
            blocks = official_blocks(p, split)
            e = np.load(f"{OUT}/{split}11F_{p}.npz", allow_pickle=True)
            names = [str(n) for n in e["names"]]
            for w, (d, T, c) in enumerate(blocks):
                cut = d * 288 + T                                   # last row stamped <= origin
                lz = Layer.__new__(Layer)
                lz.vcut, lz.sign, lz.dates = lay.vcut, lay.sign, lay.dates
                lz.el = lay.el
                def trunc(a, fillv):
                    b = a.copy().reshape(-1, a.shape[-1]); b[cut + 1:] = fillv
                    return b.reshape(a.shape)
                lz.Q = trunc(lay.Q, -1); lz.R = trunc(lay.R, np.nan)
                lz.ron = trunc(lay.ron, np.nan); lz.roff = trunc(lay.roff, np.nan)
                Qc = Q0.copy()
                Qc[off:off + len(lay.dates)] = lz.Q
                earlier = [b for b in blocks if b[0] * 288 + b[1] < cut]
                Qf = fill_blocks(Qc, lay.el, lz, earlier, m, day_offset=off)
                cols, F = calendar_columns(Qf, cal, off + d, T, names)
                eq = np.array_equal(np.nan_to_num(F, nan=-7), np.nan_to_num(e["X"][w][:, cols], nan=-7))
                same += int(eq); total += 1
        print(p, "identical so far", same, "/", total, flush=True)
    print(f"official windows with identical month features after deleting every row after T: {same} / {total}")


if __name__ == "__main__":
    {"fit": fit, "calendar": calendar, "month": month, "test": deletion_test}[sys.argv[1]]()
