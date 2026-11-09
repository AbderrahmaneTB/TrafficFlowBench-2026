"""Task 1 features: every target cell is described by what the masked layer shows
around it in time (same link) and space (neighbouring links, same moment)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import RELEASE, WORK, PANELS, load, network, free_speed

REG = {"R1": 0, "R2": 1, "R3": 2}
K = 3  # temporal offsets
J = 3  # spatial offsets
# feature version: 1 = original; 2 = profiles from the target's own month (masked layer) instead of
# the train-wide ones (validation/private run on a new scenario seed), plus +-30/60 min offsets;
# 3 = 2 + a low-rank completion of the month (t1_lowrank) and the neighbours' deviations from it;
# 4 = 3 with the link free-flow speed taken from the target's own month: every new scenario redraws each link's
#     free-flow level (by ~1 km/h, within the train scenario it moves by 0.03), and the train-wide level biased
#     free-flow cells in March/April by most of that shift;
# 5 = 4 + the released on-ramp flows at the same moment (Task 1 is offline): an on-ramp shuts off in the 5-minute
#     step its mainline link jams, which flags the jam edges that the mainline neighbours leave ambiguous;
# 6 = 5 + a second, finer month fit (rank 12) and the time neighbours' deviation from it;
# 7 = 6 + month fits of rank 24 and 48 (each finer fit kept adding on held-out March days and on February)
FEAT_VERSION = int(os.environ.get("TFB_T1FEAT", "1"))


def onramp_ratio(z: dict) -> np.ndarray:
    """(D, 288, L) smallest flow ratio (flow / the ramp's mean at that slot over the layer's days) of the on-ramps
    attached to each link; NaN where a link has no observed on-ramp."""
    links = {str(x): i for i, x in enumerate(z["links"])}
    rl = np.array([links.get(str(x), -1) for x in z["ramp_link"]])
    on = np.flatnonzero((np.array([str(x) for x in z["ramp_type"]]) == "OR") & (rl >= 0))
    rq = np.where((z["relig"][..., on] > 0) & np.isfinite(z["rq"][..., on]), z["rq"][..., on], np.nan)
    with np.errstate(all="ignore"):
        ratio = rq / np.maximum(np.nanmean(rq, axis=0), 20.0)[None]
    D, T, L = z["mv"].shape
    out = np.full((D, T, L), np.nan, np.float32)
    for k, r in enumerate(on):
        a, b = out[:, :, rl[r]], ratio[:, :, k]
        out[:, :, rl[r]] = np.where(np.isnan(a), b, np.where(np.isnan(b), a, np.minimum(a, b)))
    return out


def month_free_speed(z: dict, vfh: np.ndarray):
    """(n_months, L) median observed free-flow speed per link and calendar month of the layer passed in, and the
    month index of every day; the train-wide level where a link has too few free-flow observations."""
    dates = [str(x) for x in z["dates"]]
    mon = np.array([d[:7] for d in dates])
    months = sorted(set(mon))
    el = z["melig"].astype(bool) & np.isfinite(z["mv"])
    out = []
    for m in months:
        r = mon == m
        v = np.where(el[r] & (z["mv"][r] > 0.85 * vfh[None, None, :]), z["mv"][r], np.nan).reshape(-1, len(vfh))
        n = np.isfinite(v).sum(axis=0)
        with np.errstate(all="ignore"):
            med = np.nanmedian(v, axis=0)
        out.append(np.where(n >= 50, med, vfh).astype(np.float32))
    return np.stack(out), np.array([months.index(x) for x in mon])


def month_lowrank(z: dict, lanes: np.ndarray, rank: int = 6, lam: float = 5.0):
    """(D, 288, L) low-rank flow per lane and speed, one fit per calendar month of the layer passed in
    (its blank cells, Task 1 targets or pseudo-targets, never enter the fit)."""
    from t1_lowrank import als
    dates = [str(x) for x in z["dates"]]
    mon = np.array([d[:7] for d in dates])
    D, T, L = z["mv"].shape
    el = z["melig"].astype(bool) & np.isfinite(z["mv"]) & np.isfinite(z["mq"])
    lq = np.full((D, T, L), np.nan, np.float32); lv = np.full((D, T, L), np.nan, np.float32)
    for m in sorted(set(mon)):
        r = np.flatnonzero(mon == m)
        W = el[r].reshape(-1, L)
        lq[r] = als(np.nan_to_num((z["mq"][r] / lanes).reshape(-1, L).astype(np.float64)), W, rank, lam).reshape(len(r), T, L)
        lv[r] = als(np.nan_to_num(z["mv"][r].reshape(-1, L).astype(np.float64)), W, rank, lam).reshape(len(r), T, L)
    return lq, lv


def month_profiles(z: dict, lanes: np.ndarray, vcut: np.ndarray):
    """Per calendar month of the split: (288, L) mean speed, flow per lane, queued share and count,
    from the masked layer's eligible observations (Task 1 targets are blank there)."""
    dates = [str(x) for x in z["dates"]]
    mon = np.array([d[:7] for d in dates])
    months = sorted(set(mon))
    el = z["melig"].astype(bool) & np.isfinite(z["mv"]) & np.isfinite(z["mq"])
    v = np.where(el, z["mv"], 0.0); q = np.where(el, z["mq"] / lanes, 0.0)
    out = {k: [] for k in ("v", "q", "qu", "n")}
    for m in months:
        r = mon == m
        n = el[r].sum(0)
        out["n"].append(n.astype(np.float32))
        out["v"].append(np.where(n > 0, v[r].sum(0) / np.maximum(n, 1), np.nan).astype(np.float32))
        out["q"].append(np.where(n > 0, q[r].sum(0) / np.maximum(n, 1), np.nan).astype(np.float32))
        out["qu"].append(np.where(n > 0, (el[r] & (z["mv"][r] <= vcut)).sum(0) / np.maximum(n, 1), np.nan).astype(np.float32))
    day_month = np.array([months.index(x) for x in mon])
    return {k: np.stack(a) for k, a in out.items()}, day_month


def target_index(panel: str, split: str, dates: list[str], links: list[str]) -> dict:
    path = f"{WORK}/t1idx_{panel}_{split}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    tpl = pd.read_csv(f"{RELEASE}/task1/{panel}/{split}/sample_submission_state.csv",
                      usecols=["timestamp", "station_id", "link_id", "mask_regime"], dtype=str)
    di = pd.Index(dates).get_indexer(tpl.timestamp.str[:10])
    ti = (tpl.timestamp.str[11:13].astype(int) * 12 + tpl.timestamp.str[14:16].astype(int) // 5).to_numpy()
    li = pd.Index(links).get_indexer(tpl.link_id)
    assert (di >= 0).all() and (li >= 0).all()
    out = dict(d=di.astype(np.int16), t=ti.astype(np.int16), l=li.astype(np.int16),
               reg=tpl.mask_regime.map(REG).to_numpy(np.int8))
    np.savez(path, **out)
    return out


def profiles(panel: str) -> dict:
    """Per link and 5-minute slot: mean speed, mean flow per lane, queued share (train, unmasked)."""
    path = f"{WORK}/profile_{panel}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    z = load(panel, "train")
    net = network(panel)
    lanes = net.lanes.to_numpy(np.float32)
    v = np.where(z["elig"].astype(bool), z["v"], np.nan)
    q = np.where(z["elig"].astype(bool), z["q"], np.nan) / lanes
    out = dict(v=np.nanmean(v, axis=0).astype(np.float32), q=np.nanmean(q, axis=0).astype(np.float32),
               qu=np.nanmean(np.where(np.isfinite(v), v <= net.vcut.to_numpy(), np.nan), axis=0).astype(np.float32),
               vsd=np.nanstd(v, axis=0).astype(np.float32))
    np.savez(path, **out)
    return out


def _prev_next(finite: np.ndarray):
    """Index of the previous / next finite slot along axis 1 (time), -1 / big when none."""
    D, T, L = finite.shape
    pos = np.arange(T, dtype=np.int32)[None, :, None]
    a = np.where(finite, pos, -1)
    prev = np.maximum.accumulate(a, axis=1)
    prev = np.concatenate([np.full((D, 1, L), -1, np.int32), prev[:, :-1]], axis=1)
    b = np.where(finite, pos, 10_000)
    nxt = np.minimum.accumulate(b[:, ::-1], axis=1)[:, ::-1]
    nxt = np.concatenate([nxt[:, 1:], np.full((D, 1, L), 10_000, np.int32)], axis=1)
    return prev, nxt


def _near_space(finite: np.ndarray):
    """Index of the nearest finite link below / above along axis 2 (milepost order)."""
    D, T, L = finite.shape
    pos = np.arange(L, dtype=np.int32)[None, None, :]
    a = np.where(finite, pos, -1)
    lo = np.maximum.accumulate(a, axis=2)
    lo = np.concatenate([np.full((D, T, 1), -1, np.int32), lo[:, :, :-1]], axis=2)
    b = np.where(finite, pos, 10_000)
    hi = np.minimum.accumulate(b[:, :, ::-1], axis=2)[:, :, ::-1]
    hi = np.concatenate([hi[:, :, 1:], np.full((D, T, 1), 10_000, np.int32)], axis=2)
    return lo, hi


def build(panel: str, split: str, z: dict | None = None, sel: np.ndarray | None = None, idx: dict | None = None):
    """Return (X, names, base_v, base_q_lane, idx, truth or None) for the split's targets
    (or for the cells of an explicit idx, e.g. self-supervised pseudo-targets)."""
    z = z if z is not None else load(panel, split)
    net = network(panel)
    dates = [str(x) for x in z["dates"]]
    links = [str(x) for x in z["links"]]
    idx = idx if idx is not None else target_index(panel, split, dates, links)
    if sel is not None:
        idx = {k: v[sel] for k, v in idx.items()}
    d, t, l = idx["d"].astype(np.int64), idx["t"].astype(np.int64), idx["l"].astype(np.int64)
    lanes = net.lanes.to_numpy(np.float32)
    cap = net.capacity_vph.to_numpy(np.float32) / lanes
    vfh = free_speed(panel)
    prof = profiles(panel)
    D, T, L = z["mv"].shape
    if FEAT_VERSION >= 4:
        mvf, dmv = month_free_speed(z, vfh)
        vfd = mvf[dmv]                      # (D, L) free-flow level of each day's own month
    else:
        vfd = np.broadcast_to(vfh, (D, L))
    V = z["mv"]
    Q = z["mq"] / lanes
    Kd = z["mo"] * 285.0 / lanes  # occupancy density per lane
    P = z["mpct"].astype(np.float32)
    fin = np.isfinite(V) & np.isfinite(Q)
    prev, nxt = _prev_next(fin)
    lo, hi = _near_space(fin)
    feats, names = [], []

    def add(name, arr):
        names.append(name); feats.append(np.asarray(arr, np.float32))

    sign = net.attrs["direction"]
    add("panel", np.full(len(d), PANELS.index(panel)))
    add("link_pos", l / max(L - 1, 1))
    add("slot", t)
    dow = pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()
    add("dow", dow[d])
    add("regime", idx["reg"])
    add("vf", vfd[d, l]); add("lanes", lanes[l]); add("cap_lane", cap[l])
    add("length", net.length_km.to_numpy()[l])
    add("pct", P[d, t, l])
    if FEAT_VERSION >= 2:
        mprof, dm = month_profiles(z, lanes, net.vcut.to_numpy(np.float32))
        mi = dm[d]
        pv_ = mprof["v"][mi, t, l]; pq_ = mprof["q"][mi, t, l]
        add("mprof_v", pv_); add("mprof_q", pq_); add("mprof_qu", mprof["qu"][mi, t, l]); add("mprof_n", mprof["n"][mi, t, l])
        # a slot never observed in the month falls back to the train-wide profile
        prof_v_at = np.where(np.isfinite(pv_), pv_, prof["v"][t, l])
        prof_q_at = np.where(np.isfinite(pq_), pq_, prof["q"][t, l])
    else:
        add("prof_v", prof["v"][t, l]); add("prof_q", prof["q"][t, l]); add("prof_qu", prof["qu"][t, l])
        add("prof_vsd", prof["vsd"][t, l])
        prof_v_at = prof["v"][t, l]; prof_q_at = prof["q"][t, l]
    # same link, fixed offsets in time
    for k in (-3, -2, -1, 1, 2, 3):
        tt = t + k
        ok = (tt >= 0) & (tt < T)
        tt = np.clip(tt, 0, T - 1)
        add(f"v_t{k:+d}", np.where(ok, V[d, tt, l], np.nan))
        add(f"q_t{k:+d}", np.where(ok, Q[d, tt, l], np.nan))
        if abs(k) == 1:
            add(f"k_t{k:+d}", np.where(ok, Kd[d, tt, l], np.nan))
    if FEAT_VERSION >= 2:
        for k in (-12, -6, 6, 12):
            tt = t + k
            ok = (tt >= 0) & (tt < T)
            tt = np.clip(tt, 0, T - 1)
            add(f"v_t{k:+d}", np.where(ok, V[d, tt, l], np.nan))
            add(f"q_t{k:+d}", np.where(ok, Q[d, tt, l], np.nan))
    # previous / next observed on the same link
    p = prev[d, t, l]; n = nxt[d, t, l]
    hp = p >= 0; hn = n < T
    pc = np.clip(p, 0, T - 1); nc = np.clip(n, 0, T - 1)
    vp = np.where(hp, V[d, pc, l], np.nan); vn = np.where(hn, V[d, nc, l], np.nan)
    qp = np.where(hp, Q[d, pc, l], np.nan); qn = np.where(hn, Q[d, nc, l], np.nan)
    gp = np.where(hp, t - p, 99).astype(np.float32); gn = np.where(hn, n - t, 99).astype(np.float32)
    add("v_prev", vp); add("v_next", vn); add("q_prev", qp); add("q_next", qn)
    add("k_prev", np.where(hp, Kd[d, pc, l], np.nan)); add("k_next", np.where(hn, Kd[d, nc, l], np.nan))
    add("gap_prev", gp); add("gap_next", gn)
    w = gp / (gp + gn)
    both = hp & hn
    vi = np.where(both, vp + (vn - vp) * w, np.where(hp, vp, vn))
    qi = np.where(both, qp + (qn - qp) * w, np.where(hp, qp, qn))
    add("v_interp", vi); add("q_interp", qi)
    # neighbouring links at the same time, in direction of travel (u = upstream)
    for j in (1, 2, 3):
        for tag, jj in (("u", -sign * j), ("d", sign * j)):
            ll = l + jj
            ok = (ll >= 0) & (ll < L)
            ll = np.clip(ll, 0, L - 1)
            add(f"v_{tag}{j}", np.where(ok, V[d, t, ll], np.nan))
            add(f"q_{tag}{j}", np.where(ok, Q[d, t, ll], np.nan))
            if j == 1:
                add(f"k_{tag}{j}", np.where(ok, Kd[d, t, ll], np.nan))
                for k in (-1, 1):
                    tt = np.clip(t + k, 0, T - 1)
                    okt = ok & (t + k >= 0) & (t + k < T)
                    add(f"v_{tag}{j}_t{k:+d}", np.where(okt, V[d, tt, ll], np.nan))
                    add(f"q_{tag}{j}_t{k:+d}", np.where(okt, Q[d, tt, ll], np.nan))
    # nearest observed link up / down (any distance)
    a = lo[d, t, l]; b_ = hi[d, t, l]
    ha = a >= 0; hb = b_ < L
    ac = np.clip(a, 0, L - 1); bc = np.clip(b_, 0, L - 1)
    mp = net.milepost.to_numpy()
    va = np.where(ha, V[d, t, ac], np.nan); vb = np.where(hb, V[d, t, bc], np.nan)
    if sign > 0:
        add("v_nu", va); add("v_nd", vb); add("q_nu", np.where(ha, Q[d, t, ac], np.nan)); add("q_nd", np.where(hb, Q[d, t, bc], np.nan))
        add("dist_nu", np.where(ha, np.abs(mp[l] - mp[ac]), 99)); add("dist_nd", np.where(hb, np.abs(mp[bc] - mp[l]), 99))
    else:
        add("v_nu", vb); add("v_nd", va); add("q_nu", np.where(hb, Q[d, t, bc], np.nan)); add("q_nd", np.where(ha, Q[d, t, ac], np.nan))
        add("dist_nu", np.where(hb, np.abs(mp[bc] - mp[l]), 99)); add("dist_nd", np.where(ha, np.abs(mp[l] - mp[ac]), 99))
    # local congestion context: min speed ratio over +-5 links at t (observed only)
    win = []
    for j in range(-5, 6):
        ll = np.clip(l + j, 0, L - 1)
        win.append(np.where((l + j >= 0) & (l + j < L), V[d, t, ll] / vfd[d, ll], np.nan))
    win = np.stack(win)
    with np.errstate(all="ignore"):
        add("ctx_min_vr", np.nanmin(win, axis=0)); add("ctx_mean_vr", np.nanmean(win, axis=0))
        add("ctx_nobs", np.isfinite(win).sum(axis=0))
    if FEAT_VERSION >= 3:
        lq, lv = month_lowrank(z, lanes)
        add("lr_v", lv[d, t, l]); add("lr_q", lq[d, t, l])
        # how the observed neighbours deviate from the low-rank surface (jams, day-specific shifts)
        rv_, rq_ = V - lv, Q - lq
        dt_v, dt_q, ds_v, ds_q = [], [], [], []
        for k in (-2, -1, 1, 2):
            tt = np.clip(t + k, 0, T - 1); ok = (t + k >= 0) & (t + k < T)
            dt_v.append(np.where(ok, rv_[d, tt, l], np.nan)); dt_q.append(np.where(ok, rq_[d, tt, l], np.nan))
            ll = np.clip(l + k, 0, L - 1); ok = (l + k >= 0) & (l + k < L)
            ds_v.append(np.where(ok, rv_[d, t, ll], np.nan)); ds_q.append(np.where(ok, rq_[d, t, ll], np.nan))
        with np.errstate(all="ignore"):
            add("lr_dv_t", np.nanmean(dt_v, axis=0)); add("lr_dq_t", np.nanmean(dt_q, axis=0))
            add("lr_dv_s", np.nanmean(ds_v, axis=0)); add("lr_dq_s", np.nanmean(ds_q, axis=0))
    if FEAT_VERSION >= 6:
        lq12, lv12 = month_lowrank(z, lanes, 12, 10.0)
        add("lr12_v", lv12[d, t, l]); add("lr12_q", lq12[d, t, l])
        dv12, dq12 = [], []
        for k in (-2, -1, 1, 2):
            tt = np.clip(t + k, 0, T - 1); ok = (t + k >= 0) & (t + k < T)
            dv12.append(np.where(ok, V[d, tt, l] - lv12[d, tt, l], np.nan)); dq12.append(np.where(ok, Q[d, tt, l] - lq12[d, tt, l], np.nan))
        with np.errstate(all="ignore"):
            add("lr12_dv_t", np.nanmean(dv12, axis=0)); add("lr12_dq_t", np.nanmean(dq12, axis=0))
    if FEAT_VERSION >= 7:
        for rk, lam in ((24, 20.0), (48, 40.0)):
            lqk, lvk = month_lowrank(z, lanes, rk, lam)
            add(f"lr{rk}_v", lvk[d, t, l]); add(f"lr{rk}_q", lqk[d, t, l])
            dvk, dqk = [], []
            for k in (-2, -1, 1, 2):
                tt = np.clip(t + k, 0, T - 1); ok = (t + k >= 0) & (t + k < T)
                dvk.append(np.where(ok, V[d, tt, l] - lvk[d, tt, l], np.nan)); dqk.append(np.where(ok, Q[d, tt, l] - lqk[d, tt, l], np.nan))
            with np.errstate(all="ignore"):
                add(f"lr{rk}_dv_t", np.nanmean(dvk, axis=0)); add(f"lr{rk}_dq_t", np.nanmean(dqk, axis=0))
    if FEAT_VERSION >= 5:
        ron = onramp_ratio(z)
        for k in (-1, 0, 1):
            tt = np.clip(t + k, 0, T - 1); ok = (t + k >= 0) & (t + k < T)
            add(f"ron_t{k:+d}", np.where(ok, ron[d, tt, l], np.nan))
        for tag, jj in (("u1", -sign), ("d1", sign)):
            ll = np.clip(l + jj, 0, L - 1); ok = (l + jj >= 0) & (l + jj < L)
            add(f"ron_{tag}", np.where(ok, ron[d, t, ll], np.nan))
        near = [np.where((l + j >= 0) & (l + j < L), ron[d, t, np.clip(l + j, 0, L - 1)], np.nan) for j in range(-3, 4)]
        with np.errstate(all="ignore"):
            add("ron_min3", np.nanmin(np.stack(near), axis=0))
    X = np.column_stack(feats).astype(np.float32)
    # base predictions the models correct
    base_v = np.where(np.isfinite(vi) & (gp + gn <= 24), vi, np.nan)
    base_v = np.where(np.isfinite(base_v), base_v, prof_v_at)
    base_v = np.where(np.isfinite(base_v), base_v, vfd[d, l])
    base_q = np.where(np.isfinite(qi) & (gp + gn <= 24), qi, np.nan)
    base_q = np.where(np.isfinite(base_q), base_q, prof_q_at)
    truth = None
    if "v" in z:
        truth = (z["v"][d, t, l], z["q"][d, t, l] / lanes[l])
    return X, names, base_v.astype(np.float32), base_q.astype(np.float32), idx, truth
