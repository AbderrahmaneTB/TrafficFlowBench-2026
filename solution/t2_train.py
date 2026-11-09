"""Train the Task 2 queue models and tune the decoding on held-out February windows."""
from __future__ import annotations

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

from common import WORK, QUEUE_PANELS

OUT = f"{WORK}/t2"
CAND = 30


class CatModel:
    """A CatBoost classifier behind the calls the Task 2 code makes on a LightGBM Booster."""

    def __init__(self, m):
        self.m = m

    def predict(self, X):
        return self.m.predict(X, prediction_type="Probability")[:, 1]

    def num_feature(self):
        return len(self.m.feature_names_)

    def feature_importance(self, kind="gain"):
        return self.m.get_feature_importance()

    @property
    def best_iteration(self):
        return self.m.get_best_iteration() + 1


def load_model(kind: str, tag: str, full: bool = False):
    """The ongoing / onset model of a tag: LightGBM (.txt) or CatBoost (.cbm)."""
    base = f"{OUT}/{kind}_{tag}{'_full' if full else ''}"
    if os.path.exists(base + ".cbm"):
        from catboost import CatBoostClassifier
        return CatModel(CatBoostClassifier().load_model(base + ".cbm"))
    return lgb.Booster(model_file=base + ".txt")


def fit_cat(Xa, ya, wa, Xb, yb, wb, names, rounds, full, seed, device):
    """CatBoost on the same rows and weights as the LightGBM models; early stopping on the February rows."""
    from catboost import CatBoostClassifier, Pool
    kw = dict(iterations=rounds, learning_rate=0.06, depth=8, l2_leaf_reg=10, border_count=254,
              loss_function="Logloss", random_seed=seed, verbose=200, allow_writing_files=False)
    if device != "CPU":
        kw.update(task_type="GPU", devices=device)
    if not full:
        kw.update(od_type="Iter", od_wait=200, use_best_model=True)
    m = CatBoostClassifier(**kw)
    m.fit(Pool(Xa, ya, weight=wa, feature_names=names), eval_set=None if full else Pool(Xb, yb, weight=wb, feature_names=names))
    return CatModel(m)


MAX_ONGOING = 550   # ongoing training windows kept per panel (memory)
AUG_COLS = ["load_mean3", "dens_mean3", "load_last", "dens_last", "dens_trend", "load_trend",
            "head_rate", "prof_qu6", "vr_mean3", "vr_min13", "dens_mean13", "load_mean13",
            # v6 month-own activity; absent from older layouts, which skip them
            "mq_h6", "mq_dayany", "mq_allday", "sd_share"]


def augment(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Window-relative features: how a link compares with the rest of the corridor at T.

    X is one window (L, F). Onset is a competition between bottlenecks, which a per-link
    model cannot see without these."""
    ci = {n: i for i, n in enumerate(names)}
    extra, new = [], []
    L = X.shape[0]
    for c in AUG_COLS:
        if c not in ci:
            continue
        v = X[:, ci[c]].astype(np.float64)
        ok = np.isfinite(v)
        r = np.full(L, np.nan)
        if ok.any():
            order = np.argsort(np.where(ok, v, -np.inf))
            rk = np.empty(L); rk[order] = np.arange(L)
            r = np.where(ok, (rk - (L - ok.sum())) / max(ok.sum() - 1, 1), np.nan)
            mx, mn = np.nanmax(v), np.nanmin(v)
            sd = np.nanstd(v) + 1e-6
            extra += [r, v - mx, (v - np.nanmean(v)) / sd]
        else:
            extra += [r, r, r]
        new += [f"{c}_rank", f"{c}_dmax", f"{c}_z"]
    return np.column_stack([X] + extra).astype(np.float32), names + new


CORR_NAMES = ["c_dens", "c_dens_trend", "c_dens_dev13", "c_q_km", "c_q_share",
              "dens_n+1", "dens_n-1", "dens_up3km", "dens_dn3km"]
# level 2: more spatial scales, and how each link's surroundings rank within the window (bottleneck competition)
CORR2_NAMES = ["dens_up1km", "dens_dn1km", "dens_up6km", "dens_dn6km", "load_up3km", "load_dn3km",
               "dtr_up3km", "dtr_dn3km"] + [f"{c}_{s}" for c in ("dens_up3km", "dens_dn3km", "load_dn3km", "dens_dn1km")
                                            for s in ("rank", "z")]


# level 3: density per lane (veh/km/lane): equal to the critical-density ratio up to a per-link constant, but between
# links it differs exactly where lanes or capacity change, i.e. at the bottlenecks
CORR3_NAMES = ["klane", "klane_n+1", "klane_n-1", "klane_up1km", "klane_dn1km", "klane_up3km", "klane_dn3km",
               "klane_dn1km_rank", "klane_dn1km_z", "klane_ahead"]


def corr_names(level: int) -> list[str]:
    return CORR_NAMES + (CORR2_NAMES if level >= 2 else []) + (CORR3_NAMES if level >= 3 else [])


def _rank_z(v):
    ok = np.isfinite(v)
    if not ok.any():
        return np.full(len(v), np.nan), np.full(len(v), np.nan)
    order = np.argsort(np.where(ok, v, -np.inf))
    rk = np.empty(len(v)); rk[order] = np.arange(len(v))
    r = np.where(ok, (rk - (len(v) - ok.sum())) / max(ok.sum() - 1, 1), np.nan)
    return r, (v - np.nanmean(v)) / (np.nanstd(v) + 1e-6)


def corridor(X: np.ndarray, names: list[str], level: int = 1) -> np.ndarray:
    """Corridor-level state of one window (L, F) at T, and the density around each link in travel order.

    The number of vehicles on a corridor sets the length of its jams, and the density just downstream of a
    bottleneck is the earliest precursor of a breakdown there. Built only from features that are themselves
    computed from rows at or before T."""
    ci = {n: i for i, n in enumerate(names)}
    L = X.shape[0]
    col = lambda n: X[:, ci[n]].astype(np.float64)
    length = col("length")
    w = col("cap_lane") * col("lanes") / np.maximum(col("vf"), 1.0) * length     # critical vehicles per link

    def wmean(v):
        ok = np.isfinite(v)
        return float(np.sum(v[ok] * w[ok]) / max(np.sum(w[ok]), 1e-9)) if ok.any() else np.nan

    d = col("dens_last")
    c_d = wmean(d)
    qkm = float(length[np.nan_to_num(col("q_last")) > 0.5].sum())
    out = [np.full(L, c_d), np.full(L, wmean(col("dens_trend"))), np.full(L, c_d - wmean(col("dens_mean13"))),
           np.full(L, qkm), np.full(L, qkm / max(length.sum(), 1e-9))]
    order = np.argsort(col("link_pos"), kind="stable")                 # travel order, most upstream first
    lo = length[order]
    x = np.cumsum(lo) - lo / 2
    i = np.arange(L)

    def around(v, km):
        """Mean of v (travel order) over the links within km upstream / downstream of each link, itself excluded."""
        fin = np.isfinite(v)
        cs = np.r_[0.0, np.cumsum(np.where(fin, v, 0.0))]; cn = np.r_[0, np.cumsum(fin)]
        j0 = np.searchsorted(x, x - km, "left"); j1 = np.searchsorted(x, x + km, "right")
        with np.errstate(invalid="ignore", divide="ignore"):
            return (cs[i] - cs[j0]) / (cn[i] - cn[j0]), (cs[j1] - cs[i + 1]) / (cn[j1] - cn[i + 1])

    do = d[order]
    up, dn = around(do, 3.0)
    per = [np.r_[do[1:], np.nan], np.r_[np.nan, do[:-1]], up, dn]
    if level >= 2:
        up1, dn1 = around(do, 1.0); up6, dn6 = around(do, 6.0)
        lup, ldn = around(col("load_last")[order], 3.0)
        tup, tdn = around(col("dens_trend")[order], 3.0)
        per += [up1, dn1, up6, dn6, lup, ldn, tup, tdn]
        for v in (up, dn, ldn, dn1):
            per += list(_rank_z(v))
    if level >= 3:
        kl = (d * col("cap_lane") / np.maximum(col("vf"), 1.0))[order]
        kup1, kdn1 = around(kl, 1.0); kup3, kdn3 = around(kl, 3.0)
        per += [kl, np.r_[kl[1:], np.nan], np.r_[np.nan, kl[:-1]], kup1, kdn1, kup3, kdn3]
        per += list(_rank_z(kdn1))
        per += [kdn1 - kl]                                  # filling ahead: downstream denser than here
    rank = np.empty(L, int); rank[order] = i
    out += [a[rank] for a in per]
    return np.column_stack(out).astype(np.float32)


KIN_NAMES = ["kin_s", "kin_dx", "kin_t", "kin_qa", "kin_qj", "kin_dh", "kin_hvt"]


def kinematic(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Kinematic-wave (LWR) state of the queues of one window (L, F) at T, per link (Nov 6, Fable idea S3).

    For every queued cluster at T (the same clusters as t2_data: state = last observation within 3 steps, unknown
    gaps of one link allowed) the tail is a shock between the arriving traffic (up to 3 unqueued links within 1.5 km
    upstream of the tail) and the queue: s = vf * (load_jam - load_arr) / (dens_jam - dens_arr) in km/h, with load =
    q / capacity and dens = k / k_crit (so capacity / k_crit = vf); s < 0: the tail moves upstream (the queue grows).
    Per link, for its own cluster or else the nearest cluster downstream: kin_s, kin_dx = km to the tail (> 0
    upstream of it, < 0 inside), kin_t = minutes until the tail reaches the link (upstream, s < 0) or passes it
    (inside, s > 0), else 999, kin_qa / kin_qj = arrival and queue loads, kin_dh = km to the cluster's head,
    kin_hvt = speed-ratio trend of the head link (a recovering head precedes dissolution).
    Built only from features of rows at or before T; must be computed before the dropped columns are blanked
    (it orders links with link_pos)."""
    ci = {n: i for i, n in enumerate(names)}
    col = lambda n: X[:, ci[n]].astype(np.float64)
    L = X.shape[0]
    out = np.full((L, len(KIN_NAMES)), np.nan)
    order = np.argsort(col("link_pos"), kind="stable")                 # travel order, most upstream first
    lo = col("length")[order]
    x = np.cumsum(lo) - lo / 2
    load, dens, vf = col("load_last")[order], col("dens_last")[order], col("vf")[order]
    vrt = col("vr_trend")[order]
    st = np.where(col("age")[order] <= 3, col("q_last")[order], np.nan)
    filled = np.nan_to_num(st, nan=0.0)
    clusters, i = [], 0
    while i < L:
        if filled[i] == 1:
            j = i
            while j + 1 < L and (filled[j + 1] == 1 or (np.isnan(st[j + 1]) and j + 2 < L and filled[j + 2] == 1)):
                j += 1
            clusters.append((i, j)); i = j + 1
        else:
            i += 1
    if not clusters:
        return out.astype(np.float32)
    info = []
    for a, b in clusters:
        up = [k for k in range(a - 1, max(a - 4, -1), -1)
              if x[a] - x[k] <= 1.5 and filled[k] == 0 and np.isfinite(load[k]) and np.isfinite(dens[k])]
        jam = [k for k in range(a, b + 1) if np.isfinite(load[k]) and np.isfinite(dens[k])]
        qa = float(np.mean(load[up])) if up else np.nan
        ka = float(np.mean(dens[up])) if up else np.nan
        qj = float(np.mean(load[jam])) if jam else np.nan
        kj = float(np.mean(dens[jam])) if jam else np.nan
        s = np.nan
        if up and jam and kj - ka > 0.05:
            s = float(np.clip(vf[a] * (qj - qa) / (kj - ka), -40.0, 40.0))
        info.append((a, b, s, qa, qj, vrt[b]))
    starts = np.array([c[0] for c in info])
    res = np.full((L, len(KIN_NAMES)), np.nan)
    for k in range(L):
        inside = [c for c in info if c[0] <= k <= c[1]]
        if inside:
            a, b, s, qa, qj, hv = inside[0]
            dx = -(x[k] - x[a])
            t = (x[k] - x[a]) / s * 60 if (np.isfinite(s) and s > 0) else 999.0
        else:
            nxt = np.flatnonzero(starts > k)
            if not len(nxt):
                continue
            a, b, s, qa, qj, hv = info[nxt[0]]
            dx = x[a] - x[k]
            t = dx / (-s) * 60 if (np.isfinite(s) and s < 0) else 999.0
        res[k] = [s, dx, min(t, 999.0), qa, qj, x[b] - x[k], hv]
    out[order] = res
    return out.astype(np.float32)


def add_kin(X: np.ndarray, names) -> tuple[np.ndarray, list[str]]:
    """(W, L, F) or (L, F) -> the same with the kinematic columns appended."""
    names = list(names)
    if X.ndim == 2:
        return np.column_stack([X, kinematic(X, names)]), names + KIN_NAMES
    return np.concatenate([X, np.stack([kinematic(x, names) for x in X])], axis=2), names + KIN_NAMES


def add_corridor(X: np.ndarray, names, level: int = 1) -> tuple[np.ndarray, list[str]]:
    """(W, L, F) or (L, F) -> the same with the corridor columns of that level appended."""
    names = list(names)
    if X.ndim == 2:
        return np.column_stack([X, corridor(X, names, level)]), names + corr_names(level)
    return np.concatenate([X, np.stack([corridor(x, names, level) for x in X])], axis=2), names + corr_names(level)


AUG = False
CORR = False
KIN = False        # kinematic-wave queue-tail columns (KIN_NAMES), appended after the corridor columns
RECENT = False
# APRIL models only: emulated March windows (a new scenario seed, all of it before every April origin) join the fit
# set of these panels, repeated MARCH_REP times. Never used for a March window.
ADD_MARCH: list[str] = []
MARCH_REP = 1
DATA = ""          # "" -> train_{panel}.npz (v1/v2 windows), "3" -> train3_{panel}.npz (official selector)
RECENT_NAMES = ["rq7_now", "rq7_h6", "rq7_dayany"]
DROP: list[str] = []   # features blanked to NaN (e.g. train-period location priors that shift between months)
LOCATION_PRIORS = ["head_rate", "prof_qu0", "prof_qu3", "prof_qu6", "link_pos", "panel"]
SHIFTABLE = ["head_rate", "prof_qu0", "prof_qu3", "prof_qu6", "rq7_now", "rq7_h6", "rq7_dayany"]
PRIOR_SHIFT = 0.0   # share of training windows whose location priors are rolled to a wrong place


def shift_priors(X, names, rng):
    """Domain randomization for scenario shift: move the location priors of one window
    (L, F) by a random offset along the corridor, keeping their shape but not their place."""
    ci = {n: i for i, n in enumerate(names)}
    cols = [ci[n] for n in SHIFTABLE if n in ci]
    L = X.shape[0]
    X = X.copy()
    X[:, cols] = np.roll(X[:, cols], int(rng.integers(max(L // 4, 1), max(3 * L // 4, 2))), axis=0)
    return X


def blank(X, names, drop):
    """Set the dropped features to NaN in place of removing them, so every layout stays the same."""
    if drop:
        ci = {n: i for i, n in enumerate(names)}
        X = X.copy()
        X[..., [ci[n] for n in drop]] = np.nan
    return X


def prep(X, names):
    return augment(X, names) if AUG else (X, names)


def load_all(max_ongoing=MAX_ONGOING, seed=0):
    """Per panel: all onset windows, a capped sample of ongoing fit windows, all held-out windows."""
    parts = []
    rng = np.random.default_rng(seed)
    for p in QUEUE_PANELS:
        z = dict(np.load(f"{OUT}/train{DATA}_{p}.npz", allow_pickle=True))
        if RECENT:
            R = np.load(f"{OUT}/recent_train_{p}.npz")["R"].astype(np.float32)
            z["X"] = np.concatenate([z["X"], R], axis=2)
            z["names"] = np.array(list(z["names"]) + RECENT_NAMES)
        if CORR:
            z["X"], nm = add_corridor(z["X"], z["names"], CORR)
            z["names"] = np.array(nm)
        if KIN:
            z["X"], nm = add_kin(z["X"], z["names"])
            z["names"] = np.array(nm)
        z["X"] = blank(z["X"], list(z["names"]), DROP)
        cond, hold = z["meta_cond"], z["meta_hold"]
        on_fit = np.flatnonzero((cond == "queue_ongoing") & ~hold)
        if len(on_fit) > max_ongoing:
            on_fit = np.sort(rng.choice(on_fit, max_ongoing, replace=False))
        keep = np.sort(np.concatenate([on_fit, np.flatnonzero((cond == "queue_onset") | hold)]))
        part = {k: z[k][keep] for k in ("X", "y", "m", "meta_cond", "meta_hold", "meta_date")}
        part["names"] = z["names"]; part["panel"] = p
        if p in ADD_MARCH:
            zm = dict(np.load(f"{OUT}/validationM_{p}.npz", allow_pickle=True))
            Xm = zm["X"]
            if CORR:
                Xm, _ = add_corridor(Xm, zm["names"], CORR)
            Xm = blank(Xm, list(z["names"]), DROP)
            rep = lambda a: np.concatenate([a] * MARCH_REP)
            part["X"] = np.concatenate([part["X"], rep(Xm)]); part["y"] = np.concatenate([part["y"], rep(zm["y"])])
            part["m"] = np.concatenate([part["m"], rep(zm["m"])])
            part["meta_cond"] = np.concatenate([part["meta_cond"], rep(zm["meta_cond"])])
            part["meta_hold"] = np.concatenate([part["meta_hold"], np.zeros(len(Xm) * MARCH_REP, bool)])
            part["meta_date"] = np.concatenate([part["meta_date"], rep(zm["meta_date"])])
        if PRIOR_SHIFT > 0:
            names0 = list(z["names"])
            for w in np.flatnonzero(~part["meta_hold"]):
                if rng.random() < PRIOR_SHIFT:
                    part["X"][w] = shift_priors(part["X"][w], names0, rng)
        if AUG:
            names0 = list(z["names"])
            aug = [augment(x, names0) for x in part["X"]]
            part["X"] = np.stack([a_[0] for a_ in aug]); part["names"] = np.array(aug[0][1])
        parts.append(part)
        del z
    return parts


def candidates(X, names):
    c = {n: i for i, n in enumerate(names)}
    return (X[..., c["dist_q_down"]] <= CAND) | (X[..., c["dist_q_up"]] <= CAND) | (X[..., c["q_hist_n"]] > 0)


def ongoing_rows(parts, names, use_hold):
    Xs, ys, ws, hs = [], [], [], []
    for z in parts:
        sel = (z["meta_cond"] == "queue_ongoing") & (z["meta_hold"] == use_hold)
        X, y, m = z["X"][sel], z["y"][sel], z["m"][sel]
        cand = candidates(X, names)
        for w in range(len(X)):
            c = cand[w]
            for h in range(6):
                mm = c & m[w, h]
                if not mm.any():
                    continue
                Xs.append(np.column_stack([X[w][mm], np.full(mm.sum(), h + 1, np.float32)]))
                ys.append(y[w, h][mm])
                ws.append(np.full(mm.sum(), 1.0 / max(m[w].sum(), 1), np.float32))
    return np.vstack(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(ws) * 1000


def onset_rows(parts, use_hold):
    Xs, ys, ws = [], [], []
    for z in parts:
        sel = (z["meta_cond"] == "queue_onset") & (z["meta_hold"] == use_hold)
        X, y, m = z["X"][sel], z["y"][sel], z["m"][sel]
        for w in range(len(X)):
            mm = m[w, 5]
            Xs.append(X[w][mm]); ys.append(y[w, 5][mm])
            ws.append(np.full(mm.sum(), 1.0 / max(mm.sum(), 1), np.float32))
    return np.vstack(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(ws) * 1000


def iou(pred, y, m):
    p = pred & m; t = y & m
    u = (p | t).sum()
    return 1.0 if u == 0 else (p & t).sum() / u


def expected_iou_set(p):
    """Prefix of links sorted by probability that maximizes sum_S p / (|S| + sum_notS p)."""
    o = np.argsort(-p)
    cs = np.cumsum(p[o]); tot = p.sum()
    k = np.arange(1, len(p) + 1)
    val = cs / (k + tot - cs)
    best = int(np.argmax(val)) + 1
    sel = np.zeros(len(p), bool); sel[o[:best]] = True
    return sel


def evaluate(parts, names, m_on, m_onset, thr_on, thr_onset, mode_onset="thr", k_min=1, mode_on="thr",
             cache=None):
    res = {"queue_ongoing": [], "queue_onset": []}
    base = {"queue_ongoing": [], "queue_onset": []}
    ci = {n: i for i, n in enumerate(names)}
    cache = {} if cache is None else cache
    for zi, z in enumerate(parts):
        sel = np.flatnonzero(z["meta_hold"])
        for w in sel:
            X, y, m, cond = z["X"][w], z["y"][w], z["m"][w], z["meta_cond"][w]
            L = X.shape[0]
            pred = np.zeros((6, L), bool)
            if cond == "queue_ongoing":
                cand = candidates(X, names)
                if cand.any():
                    if (zi, w) not in cache:
                        Xc = X[cand]
                        cache[(zi, w)] = np.stack([m_on.predict(np.column_stack([Xc, np.full(len(Xc), h + 1, np.float32)]))
                                                   for h in range(6)])
                    P = cache[(zi, w)]
                    if mode_on == "thr":
                        pred[:, cand] = P >= thr_on
                    else:
                        pred[:, cand] = expected_iou_set(P.ravel()).reshape(P.shape)
                st = (X[:, ci["q_last"]] > 0.5) & (X[:, ci["age"]] <= 3)
                base[cond].append(iou(np.broadcast_to(st, (6, L)), y, m))
            else:
                if (zi, w) not in cache:
                    cache[(zi, w)] = m_onset.predict(X)
                p = cache[(zi, w)]
                if mode_onset == "thr":
                    s = p >= thr_onset
                    if s.sum() < k_min:
                        s = np.zeros(L, bool); s[np.argsort(-p)[:k_min]] = True
                else:
                    s = expected_iou_set(p)
                pred[5] = s
                base[cond].append(0.0)
            res[cond].append(iou(pred, y, m))
    return {k: float(np.mean(v)) for k, v in res.items()}, {k: float(np.mean(v)) for k, v in base.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--recent", action="store_true")
    ap.add_argument("--corr", type=int, nargs="?", const=1, default=0,
                    help="add corridor vehicle-count and neighbour-density features (level 1 = v13, 2 = more scales)")
    ap.add_argument("--kin", action="store_true", help="add the kinematic-wave queue-tail columns (KIN_NAMES)")
    ap.add_argument("--data", default="")
    ap.add_argument("--add-march", nargs="*", default=[], help="APRIL models only: panels ('all') whose emulated "
                    "March windows join the fit set")
    ap.add_argument("--march-rep", type=int, default=1)
    ap.add_argument("--learner", choices=["lgb", "cat"], default="lgb")
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min-leaf", type=int, default=50)
    ap.add_argument("--ff", type=float, default=0.7, help="LightGBM feature_fraction")
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--device", default="0", help="CatBoost: GPU devices ('0', '0:1') or CPU")
    ap.add_argument("--max-ongoing", type=int, default=MAX_ONGOING)
    ap.add_argument("--drop", nargs="*", default=[], help="feature names to blank; 'loc' = LOCATION_PRIORS")
    ap.add_argument("--prior-shift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="seed of the window sampling, prior shifts and LightGBM")
    ap.add_argument("--keep-head", action="store_true", help="do not shift head_rate: jam heads are physical "
                    "bottlenecks, the same road in every month; only the time-of-day priors are scenario-specific")
    a = ap.parse_args()
    global AUG, CORR, KIN, RECENT, DATA, DROP, PRIOR_SHIFT, ADD_MARCH, MARCH_REP
    PRIOR_SHIFT = a.prior_shift
    KIN = a.kin
    ADD_MARCH = list(QUEUE_PANELS) if a.add_march == ["all"] else a.add_march
    MARCH_REP = a.march_rep
    if a.keep_head:
        SHIFTABLE.remove("head_rate")
    AUG, CORR, RECENT, DATA = a.aug, a.corr, a.recent, a.data
    DROP = [n for d in a.drop for n in (LOCATION_PRIORS if d == "loc" else [d])]
    cfg = {"aug": AUG, "recent": RECENT, "data": DATA, "drop": DROP, "prior_shift": PRIOR_SHIFT,
           "shifted": list(SHIFTABLE) if PRIOR_SHIFT else []}
    if CORR:
        cfg["corr"] = CORR
    if KIN:
        cfg["kin"] = True
    if ADD_MARCH:
        cfg["add_march"] = ADD_MARCH; cfg["march_rep"] = MARCH_REP
    if a.learner != "lgb":
        cfg["learner"] = a.learner
    if (a.leaves, a.min_leaf, a.ff, a.l2) != (63, 50, 0.7, 10.0):
        cfg["lgb"] = dict(leaves=a.leaves, min_leaf=a.min_leaf, ff=a.ff, l2=a.l2)
    json.dump(cfg, open(f"{OUT}/config_{a.tag}.json", "w"))
    parts = load_all(a.max_ongoing, seed=a.seed)
    names = list(parts[0]["names"])
    params = dict(objective="binary", learning_rate=0.05, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf,
                  feature_fraction=a.ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=a.l2,
                  num_threads=10, verbosity=-1, seed=11 + a.seed)
    Xa, ya, wa = ongoing_rows(parts, names, False)
    Xb, yb, wb = ongoing_rows(parts, names, True)
    print("ongoing rows", Xa.shape, ya.mean(), "hold", Xb.shape, flush=True)
    if a.full:
        Xa, ya, wa = np.vstack([Xa, Xb]), np.concatenate([ya, yb]), np.concatenate([wa, wb])
    rounds = json.load(open(f"{OUT}/rounds_{a.tag}.json"))["ongoing"] if a.full else 4000
    if a.learner == "cat":
        m_on = fit_cat(Xa, ya, wa, Xb, yb, wb, names + ["h"], rounds, a.full, a.seed, a.device)
    else:
        d1 = lgb.Dataset(Xa, ya, weight=wa, feature_name=names + ["h"])
        d2 = lgb.Dataset(Xb, yb, weight=wb, reference=d1)
        cb = [lgb.log_evaluation(200)] + ([] if a.full else [lgb.early_stopping(200)])
        m_on = lgb.train(params, d1, num_boost_round=rounds, valid_sets=[d2], callbacks=cb)
    Xa, ya, wa = onset_rows(parts, False)
    Xb, yb, wb = onset_rows(parts, True)
    print("onset rows", Xa.shape, ya.mean(), "hold", Xb.shape, flush=True)
    if a.full:
        Xa, ya, wa = np.vstack([Xa, Xb]), np.concatenate([ya, yb]), np.concatenate([wa, wb])
    rounds = json.load(open(f"{OUT}/rounds_{a.tag}.json"))["onset"] if a.full else 4000
    if a.learner == "cat":
        m_onset = fit_cat(Xa, ya, wa, Xb, yb, wb, names, rounds, a.full, a.seed, a.device)
    else:
        d1 = lgb.Dataset(Xa, ya, weight=wa, feature_name=names)
        d2 = lgb.Dataset(Xb, yb, weight=wb, reference=d1)
        m_onset = lgb.train(params, d1, num_boost_round=rounds, valid_sets=[d2], callbacks=cb)
    sfx = "_full" if a.full else ""
    if a.learner == "cat":
        m_on.m.save_model(f"{OUT}/ongoing_{a.tag}{sfx}.cbm"); m_onset.m.save_model(f"{OUT}/onset_{a.tag}{sfx}.cbm")
    else:
        m_on.save_model(f"{OUT}/ongoing_{a.tag}{sfx}.txt"); m_onset.save_model(f"{OUT}/onset_{a.tag}{sfx}.txt")
    if not a.full:
        json.dump({"ongoing": m_on.best_iteration, "onset": m_onset.best_iteration}, open(f"{OUT}/rounds_{a.tag}.json", "w"))
        cache = {}
        for thr in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
            r, b = evaluate(parts, names, m_on, m_onset, thr, thr, cache=cache)
            print(f"thr={thr}: model {r}  persistence {b}", flush=True)
        for k in (1, 2, 3):
            r, _ = evaluate(parts, names, m_on, m_onset, 0.45, 0.5, k_min=k, cache=cache)
            print(f"onset thr 0.5 k_min={k}:", r, flush=True)
        r, _ = evaluate(parts, names, m_on, m_onset, 0.45, 0.5, mode_onset="eiou", mode_on="eiou", cache=cache)
        print("expected-IoU decoding (both):", r, flush=True)
        imp = sorted(zip(m_on.feature_importance("gain"), names + ["h"]), reverse=True)[:20]
        print("ongoing top:", [n for _, n in imp])
        imp = sorted(zip(m_onset.feature_importance("gain"), names), reverse=True)[:20]
        print("onset top:", [n for _, n in imp])


if __name__ == "__main__":
    main()
