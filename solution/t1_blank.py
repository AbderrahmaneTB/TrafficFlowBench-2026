"""Task 1 targets inside the panel-wide blanks that follow every official Task 2 origin.

The March / April layers hide every link of a queue panel for the 18 slots after each official origin T
(T+5 ... T+90 min); all those cells are Task 1 targets (~2 % of the targets) but no training row of the main
models looks like them: no neighbour in time or space is observed, and a month low-rank fit has no data in
those rows, so it falls back to each link's monthly mean there. The main models disagree by 6-10 km/h there
(0.2-0.6 km/h elsewhere).

Here such blanks are emulated where the truth is known, at selector-style origins (the slots the official
blanks follow): in the train months (truth = the underlying layer; the released history before T is
revealed, as in the official windows) and in March / April (truth = the released observations; Task 1 is
offline, so a scored month's own observations are legitimate training data, as for the pseudo-target rows).
The Task 1 features are rebuilt from the blanked layer, plus blank-specific ones (position in the blank,
window condition, corridor demand and speed interpolated across the blank along the month profile).
"""
from __future__ import annotations

import os

import numpy as np

from common import WORK, load, network
from t1_features import build, month_profiles, FEAT_VERSION

FEAT = f"{WORK}/t1feat"
BL = 18                    # blanked slots after an origin: T+1 ... T+18
HIST = 12                  # released history before an official origin (T-12 ... T-1), raw in the release
BLANK_NAMES = ["b_pos", "b_cond", "b_q_corr", "b_v_corr", "b_q_link", "b_nq0", "b_nq1"]


def real_blanks(z: dict) -> list[tuple[int, int]]:
    """(day, T) of every run of slots in which < 5 % of the panel's links are observed (the official blanks)."""
    frac = np.isfinite(z["mv"]).mean(axis=2)
    out = []
    for d in range(frac.shape[0]):
        b = frac[d] < 0.05
        t = 0
        while t < len(b):
            if b[t]:
                s = t
                while t < len(b) and b[t]:
                    t += 1
                if t - s >= 6:
                    out.append((d, s - 1))
            t += 1
    return out


def official_conditions(panel: str, split: str) -> dict:
    """{(day, T): 1 onset / 0 ongoing} of the official windows of a scored split."""
    w = np.load(f"{WORK}/t2/{split}_{panel}.npz", allow_pickle=True)
    z = load(panel, split)
    dates = [str(x) for x in z["dates"]]
    out = {}
    for o, c in zip(w["origin"], w["cond"]):
        o = str(o)
        out[(dates.index(o[:10]), int(o[11:13]) * 12 + int(o[14:16]) // 5)] = 1.0 if c == "queue_onset" else 0.0
    return out


def emu_windows(panel: str, split: str) -> list[tuple[int, str]]:
    """Selector-style windows (flat slot index, condition): train3 windows in the train months; in a scored month
    the reverse-engineered selector run on its released layer (as in t2_month_windows)."""
    if split == "train":
        m = np.load(f"{WORK}/t2/train3_{panel}.npz", allow_pickle=True)
        return [(int(i), str(c)) for i, c in zip(m["meta_i"], m["meta_cond"])]
    from t1_features import target_index
    from t2_build import flat
    from t2_build3 import chains
    from t2_underlying import fill_time, underlying_queue
    z = load(panel, split)
    vcut = network(panel).vcut.to_numpy(np.float32)
    dates = [str(x) for x in z["dates"]]
    mv, mel = flat(z["mv"]), flat(z["melig"]).astype(bool)
    idx = target_index(panel, split, dates, [str(x) for x in z["links"]])
    ft = idx["d"].astype(np.int64) * 288 + idx["t"].astype(np.int64)
    v = mv.copy(); v[ft, idx["l"]] = fill_time(mv)[ft, idx["l"]]
    wins = chains(v, mel, vcut, underlying_queue(mv, vcut), len(dates))
    return [(int(i), str(c)) for i, c, *_ in wins]


def blank_features(z: dict, zz: dict, idx: dict, origin: np.ndarray, cond: np.ndarray, lanes: np.ndarray,
                   vcut: np.ndarray) -> np.ndarray:
    """Blank-specific columns (BLANK_NAMES) for the cells of idx; zz is the layer the features see,
    origin / cond the blank's origin slot and condition per cell."""
    d, t, l = idx["d"].astype(np.int64), idx["t"].astype(np.int64), idx["l"].astype(np.int64)
    mprof, dm = month_profiles(zz, lanes, vcut)
    pq = mprof["q"][dm]                                   # (D, 288, L) month profile, flow per lane
    V, Q = zz["mv"], zz["mq"] / lanes
    vf = np.nanmedian(np.where(V > vcut[None, None], V, np.nan), axis=(0, 1))
    T, E = origin.astype(np.int64), np.minimum(origin.astype(np.int64) + BL + 1, 287)
    w = ((t - T) / (E - T)).astype(np.float32)
    with np.errstate(all="ignore"):
        # corridor demand ratio vs the month profile just before / after the blank, interpolated across it
        def corr_ratio(ts):
            r = Q[d, ts] / pq[d, ts]
            return np.nanmedian(np.where(np.isfinite(r) & (pq[d, ts] > 1), r, np.nan), axis=1)
        def corr_speed(ts):
            return np.nanmean(V[d, ts] / vf[None, :], axis=1)
        def queued(ts):
            return np.nansum(V[d, ts] <= vcut[None, :], axis=1).astype(np.float32)
        r0 = np.where(np.isfinite(corr_ratio(T)), corr_ratio(T), corr_ratio(np.maximum(T - 1, 0)))
        r1 = np.where(np.isfinite(corr_ratio(E)), corr_ratio(E), corr_ratio(np.minimum(E + 1, 287)))
        q_corr = pq[d, t, l] * ((1 - w) * np.where(np.isfinite(r0), r0, r1) + w * np.where(np.isfinite(r1), r1, r0))
        s0, s1 = corr_speed(T), corr_speed(E)
        v_corr = vf[l] * ((1 - w) * np.where(np.isfinite(s0), s0, s1) + w * np.where(np.isfinite(s1), s1, s0))
        # the cell's own link: last observed before / first after, scaled along the month profile
        fin = np.isfinite(V) & np.isfinite(Q)
        pos = np.arange(288)[None, :, None]
        prev = np.maximum.accumulate(np.where(fin, pos, -1), axis=1)
        nxt = np.flip(np.minimum.accumulate(np.flip(np.where(fin, pos, 288), axis=1), axis=1), axis=1)
        p, n = prev[d, t, l], nxt[d, t, l]
        pc, nc = np.clip(p, 0, 287), np.clip(n, 0, 287)
        ra = np.where(p >= 0, Q[d, pc, l] / pq[d, pc, l], np.nan)
        rb = np.where(n < 288, Q[d, nc, l] / pq[d, nc, l], np.nan)
        ww = np.where((p >= 0) & (n < 288), (t - p) / np.maximum(n - p, 1), 0.5)
        rl = np.where(np.isfinite(ra) & np.isfinite(rb), (1 - ww) * ra + ww * rb, np.where(np.isfinite(ra), ra, rb))
        q_link = pq[d, t, l] * rl
    return np.column_stack([t - T, cond, q_corr, v_corr, q_link, queued(T), queued(E)]).astype(np.float32)


def blank_rows(panel: str, split: str, group: int = 0, ngroups: int = 3) -> dict:
    """Features and truth of the cells of emulated blanks (one group of every ngroups-th window, kept apart by
    at least one blank + history so the blanks never merge)."""
    path = f"{FEAT}/{panel}_{split}_blank_g{group}of{ngroups}_f{FEAT_VERSION}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    z = load(panel, split)
    net = network(panel)
    lanes = net.lanes.to_numpy(np.float32); vcut = net.vcut.to_numpy(np.float32)
    D, T_, L = z["mv"].shape
    real = np.zeros((D, T_), bool)
    for d, T in (real_blanks(z) if split != "train" else []):
        real[d, max(T - HIST, 0):T + BL + 2] = True
    wins = sorted(emu_windows(panel, split))
    hide = np.zeros((D, T_), bool); org = np.full((D, T_), -1, np.int64); cnd = np.full((D, T_), np.nan, np.float32)
    last = -10 ** 9
    for k, (i, c) in enumerate(wins):
        d, T = divmod(i, 288)
        if k % ngroups != group or T < HIST or T + BL + 1 > 287 or i - last < BL + HIST + 1:
            continue
        if real[d, max(T - HIST, 0):T + BL + 2].any():
            continue
        hide[d, T + 1:T + BL + 1] = True; org[d, T + 1:T + BL + 1] = T
        cnd[d, T + 1:T + BL + 1] = 1.0 if c == "queue_onset" else 0.0
        last = i
    zz = dict(z)
    for k in ("mv", "mq", "mo"):
        a = z[k].copy(); a[hide] = np.nan; zz[k] = a
    if split == "train":
        # official windows hand out their history raw: reveal T-12 ... T-1 of every emulated blank
        hist = np.zeros((D, T_), bool)
        for d, t in zip(*np.nonzero(hide & ~np.roll(hide, 1, axis=1))):
            hist[d, max(t - 1 - HIST, 0):t - 1] = True
        ok = hist[:, :, None] & z["elig"].astype(bool) & np.isfinite(z["v"]) & np.isfinite(z["q"])
        for k, src in (("mv", "v"), ("mq", "q"), ("mo", "o")):
            zz[k] = np.where(ok, z[src], zz[k])
        zz["v"], zz["q"] = z["v"], z["q"]
        cell = hide[:, :, None] & z["elig"].astype(bool) & np.isfinite(z["v"]) & np.isfinite(z["q"])
    else:
        zz["v"], zz["q"] = z["mv"], z["mq"]
        cell = hide[:, :, None] & z["melig"].astype(bool) & np.isfinite(z["mv"]) & np.isfinite(z["mq"])
    d, t, l = np.nonzero(cell)
    idx = dict(d=d.astype(np.int16), t=t.astype(np.int16), l=l.astype(np.int16), reg=z["regime"][d].astype(np.int8))
    X, names, bv, bq, _, (tv, tq) = build(panel, split, zz, idx=idx)
    Xb = blank_features(z, zz, idx, org[d, t], cnd[d, t], lanes, vcut)
    out = dict(X=np.column_stack([X, Xb]), names=np.array(list(names) + BLANK_NAMES), bv=bv, bq=bq,
               tv=tv.astype(np.float32), tq=tq.astype(np.float32), reg=idx["reg"], d=idx["d"], t=idx["t"],
               l=idx["l"])
    os.makedirs(FEAT, exist_ok=True)
    np.savez(path, **out)
    return out


EXTRA_NAMES = (["ron_t-3", "ron_t-2", "ron_t+2", "ron_t+3", "ron_min5", "ron_min3_t-2", "ron_min3_t+2", "ron_low_share"]
               + [f"{w}_{j}" for j in ("u2", "u1", "d1", "d2") for w in ("vp", "vn", "gp", "gn")]
               + ["qd_prev", "qd_next", "qd_prev_u1", "qd_next_u1", "qd_prev_d1", "qd_next_d1"])


def extra_features(z: dict, zz: dict, idx: dict, panel: str) -> np.ndarray:
    """Second set of blank columns (EXTRA_NAMES): the on-ramp jam signal at more offsets and over the corridor, the
    edges of the blank on the neighbouring links (last observation before, first after) and queued flags at the edges.
    Ramps are never blanked, so z gives them; mainline edges come from the layer the features see (zz)."""
    from t1_features import onramp_ratio
    d, t, l = idx["d"].astype(np.int64), idx["t"].astype(np.int64), idx["l"].astype(np.int64)
    net = network(panel)
    vcut = net.vcut.to_numpy(np.float32); sign = net.attrs["direction"]
    ron = onramp_ratio(z)
    D, T_, L = ron.shape
    cols = []
    for k in (-3, -2, 2, 3):
        tt = np.clip(t + k, 0, T_ - 1)
        cols.append(np.where((t + k >= 0) & (t + k < T_), ron[d, tt, l], np.nan))
    with np.errstate(all="ignore"):
        def ron_min(tt, w):
            near = [np.where((l + j >= 0) & (l + j < L), ron[d, tt, np.clip(l + j, 0, L - 1)], np.nan) for j in range(-w, w + 1)]
            return np.nanmin(np.stack(near), axis=0)
        cols.append(ron_min(t, 5))
        cols.append(ron_min(np.clip(t - 2, 0, T_ - 1), 3)); cols.append(ron_min(np.clip(t + 2, 0, T_ - 1), 3))
        obs = np.isfinite(ron).sum(axis=2); low = (ron < 0.3).sum(axis=2)
        cols.append(np.where(obs[d, t] > 0, low[d, t] / np.maximum(obs[d, t], 1), np.nan))
    V = zz["mv"]
    fin = np.isfinite(V)
    pos = np.arange(T_)[None, :, None]
    prev = np.maximum.accumulate(np.where(fin, pos, -1), axis=1)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(fin, pos, T_), axis=1), axis=1), axis=1)
    edge = {}
    for tag, j in (("u2", -2 * sign), ("u1", -sign), ("0", 0), ("d1", sign), ("d2", 2 * sign)):
        ll = np.clip(l + j, 0, L - 1); ok = (l + j >= 0) & (l + j < L)
        p, n = prev[d, t, ll], nxt[d, t, ll]
        vp = np.where(ok & (p >= 0) & (p < t), V[d, np.clip(p, 0, T_ - 1), ll], np.nan)
        vn = np.where(ok & (n < T_) & (n > t), V[d, np.clip(n, 0, T_ - 1), ll], np.nan)
        edge[tag] = (vp, vn, ll)
        if tag != "0":
            cols += [vp, vn, np.where(np.isfinite(vp), t - p, 99).astype(np.float32),
                     np.where(np.isfinite(vn), n - t, 99).astype(np.float32)]
    for tag in ("0", "u1", "d1"):
        vp, vn, ll = edge[tag]
        cols += [np.where(np.isfinite(vp), (vp <= vcut[ll]).astype(np.float32), np.nan),
                 np.where(np.isfinite(vn), (vn <= vcut[ll]).astype(np.float32), np.nan)]
    return np.column_stack(cols).astype(np.float32)


EXTRA2_NAMES = ["jd_down_T", "jd_up_T", "jd_down_E", "jd_up_E", "jrun_T", "jrun_E", "nq_corr_T", "nq_corr_E",
                "roff", "roff_min3", "roff_t-2", "roff_t+2"]


def _ramp_ratio(z: dict, kind: str) -> np.ndarray:
    """(D, 288, L) smallest flow ratio (flow / the ramp's slot mean over the layer's days) of the ramps of one type
    (OR / FR) attached to each link; NaN where a link has no observed ramp of that type."""
    links = {str(x): i for i, x in enumerate(z["links"])}
    rl = np.array([links.get(str(x), -1) for x in z["ramp_link"]])
    sel = np.flatnonzero((np.array([str(x) for x in z["ramp_type"]]) == kind) & (rl >= 0))
    rq = np.where((z["relig"][..., sel] > 0) & np.isfinite(z["rq"][..., sel]), z["rq"][..., sel], np.nan)
    with np.errstate(all="ignore"):
        ratio = rq / np.maximum(np.nanmean(rq, axis=0), 20.0)[None]
    D, T, L = z["mv"].shape
    out = np.full((D, T, L), np.nan, np.float32)
    for k, r in enumerate(sel):
        a, b = out[:, :, rl[r]], ratio[:, :, k]
        out[:, :, rl[r]] = np.where(np.isnan(a), b, np.where(np.isnan(b), a, np.minimum(a, b)))
    return out


def extra2_features(z: dict, zz: dict, idx: dict, origin: np.ndarray, panel: str) -> np.ndarray:
    """Third set of blank columns (EXTRA2_NAMES): where the queues are at the blank's two edges relative to the cell
    (distance in links to the nearest queued link downstream / upstream, length of the queued run holding the cell,
    queued links of the corridor) and the off-ramp flow ratio around the cell. Edge states come from the layer the
    features see (zz): the last observation at or before T and the first at or after T+19 of every link."""
    d, t, l = idx["d"].astype(np.int64), idx["t"].astype(np.int64), idx["l"].astype(np.int64)
    org = origin.astype(np.int64)
    net = network(panel)
    vcut = net.vcut.to_numpy(np.float32); sign = int(net.attrs["direction"])
    V = zz["mv"]
    D, T_, L = V.shape
    fin = np.isfinite(V)
    pos = np.arange(T_)[None, :, None]
    prev = np.maximum.accumulate(np.where(fin, pos, -1), axis=1)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(fin, pos, T_), axis=1), axis=1), axis=1)
    out = np.full((len(d), len(EXTRA2_NAMES)), np.nan, np.float32)
    ar = np.arange(L)
    for (dd, T) in set(zip(d.tolist(), org.tolist())):
        rows = np.flatnonzero((d == dd) & (org == T))
        E = min(T + BL + 1, T_ - 1)
        feats = []
        for edge, ref in (("T", prev[dd, T]), ("E", nxt[dd, E])):
            ok = (ref >= 0) & (ref < T_) & (np.abs(ref - (T if edge == "T" else E)) <= 3)
            q = np.where(ok, V[dd, np.clip(ref, 0, T_ - 1), ar] <= vcut, False)
            qi = np.flatnonzero(q)
            # signed link offsets in travel direction: downstream = sign * (j - l) > 0
            if len(qi):
                off = sign * (qi[None, :] - ar[:, None])
                down = np.where(off >= 0, off, 10 ** 6).min(axis=1).astype(np.float32)
                up = np.where(off <= 0, -off, 10 ** 6).min(axis=1).astype(np.float32)
                down[down >= 10 ** 6] = 99; up[up >= 10 ** 6] = 99
            else:
                down = np.full(L, 99, np.float32); up = np.full(L, 99, np.float32)
            run = np.zeros(L, np.float32)
            s0 = 0
            while s0 < L:
                if q[s0]:
                    e0 = s0
                    while e0 < L and q[e0]:
                        e0 += 1
                    run[s0:e0] = e0 - s0; s0 = e0
                else:
                    s0 += 1
            feats.append((down, up, run, float(q.sum())))
        (dT, uT, rT, nT), (dE, uE, rE, nE) = feats
        ll = l[rows]
        out[rows, 0] = dT[ll]; out[rows, 1] = uT[ll]; out[rows, 2] = dE[ll]; out[rows, 3] = uE[ll]
        out[rows, 4] = rT[ll]; out[rows, 5] = rE[ll]; out[rows, 6] = nT; out[rows, 7] = nE
    roff = _ramp_ratio(z, "FR")
    out[:, 8] = roff[d, t, l]
    with np.errstate(all="ignore"):
        near = [np.where((l + j >= 0) & (l + j < L), roff[d, t, np.clip(l + j, 0, L - 1)], np.nan) for j in range(-3, 4)]
        out[:, 9] = np.nanmin(np.stack(near), axis=0)
    for c, k in ((10, -2), (11, 2)):
        tt = np.clip(t + k, 0, T_ - 1)
        out[:, c] = np.where((t + k >= 0) & (t + k < T_), roff[d, tt, l], np.nan)
    return out


def blank_extra2(panel: str, split: str, group: int = 0, ngroups: int = 3) -> np.ndarray:
    """extra2_features of the cached emulated rows of one group, cached next to them."""
    path = f"{FEAT}/{panel}_{split}_blank_g{group}of{ngroups}_f{FEAT_VERSION}_x2.npy"
    if os.path.exists(path):
        return np.load(path)
    rows = blank_rows(panel, split, group, ngroups)
    z, zz = emulated_layer(panel, split, rows)
    n = list(rows["names"])
    org = rows["t"].astype(np.int64) - rows["X"][:, n.index("b_pos")].astype(np.int64)
    out = extra2_features(z, zz, {k: rows[k] for k in ("d", "t", "l")}, org, panel)
    np.save(path, out)
    return out


def emulated_layer(panel: str, split: str, rows: dict) -> tuple[dict, dict]:
    """The blanked layer an emulated group's features were built from (scored splits), rebuilt from its rows."""
    z = load(panel, split)
    n = list(rows["names"])
    org = rows["t"].astype(np.int64) - rows["X"][:, n.index("b_pos")].astype(np.int64)
    zz = dict(z)
    hide = np.zeros(z["mv"].shape[:2], bool)
    hist = np.zeros_like(hide)
    for d, T in set(zip(rows["d"].astype(np.int64).tolist(), org.tolist())):
        hide[d, T + 1:T + BL + 1] = True; hist[d, max(T - HIST, 0):T] = True
    for k in ("mv", "mq", "mo"):
        a = z[k].copy(); a[hide] = np.nan; zz[k] = a
    if split == "train":                       # as in blank_rows: the history before T is revealed
        ok = hist[:, :, None] & z["elig"].astype(bool) & np.isfinite(z["v"]) & np.isfinite(z["q"])
        for k, src in (("mv", "v"), ("mq", "q"), ("mo", "o")):
            zz[k] = np.where(ok, z[src], zz[k])
    return z, zz


def blank_extra(panel: str, split: str, group: int = 0, ngroups: int = 3) -> np.ndarray:
    """extra_features of the cached emulated rows of one group (scored splits), cached next to them."""
    path = f"{FEAT}/{panel}_{split}_blank_g{group}of{ngroups}_f{FEAT_VERSION}_x1.npy"
    if os.path.exists(path):
        return np.load(path)
    rows = blank_rows(panel, split, group, ngroups)
    z, zz = emulated_layer(panel, split, rows)
    out = extra_features(z, zz, {k: rows[k] for k in ("d", "t", "l")}, panel)
    np.save(path, out)
    return out


def real_rows(panel: str, split: str) -> dict:
    """Features of the Task 1 targets inside the official blanks of a scored split (no extra hiding)."""
    from t1_features import target_index
    path = f"{FEAT}/{panel}_{split}_blankreal_f{FEAT_VERSION}.npz"
    if os.path.exists(path):
        c = np.load(path, allow_pickle=True)
        return {k: c[k] for k in c.files}
    z = load(panel, split)
    net = network(panel)
    lanes = net.lanes.to_numpy(np.float32); vcut = net.vcut.to_numpy(np.float32)
    dates = [str(x) for x in z["dates"]]
    conds = official_conditions(panel, split)
    D, T_, L = z["mv"].shape
    org = np.full((D, T_), -1, np.int64); cnd = np.full((D, T_), np.nan, np.float32)
    for d, T in real_blanks(z):
        e = T + 1
        while e < T_ and np.isfinite(z["mv"][d, e]).mean() < 0.05:
            e += 1
        org[d, T + 1:e] = T; cnd[d, T + 1:e] = conds.get((d, T), np.nan)
    ti = target_index(panel, split, dates, [str(x) for x in z["links"]])
    sel = org[ti["d"].astype(np.int64), ti["t"].astype(np.int64)] >= 0
    idx = {k: v[sel] for k, v in ti.items()}
    X, names, bv, bq, _, _ = build(panel, split, z, idx=idx)
    d, t = idx["d"].astype(np.int64), idx["t"].astype(np.int64)
    Xb = blank_features(z, z, idx, org[d, t], cnd[d, t], lanes, vcut)
    out = dict(X=np.column_stack([X, Xb]), names=np.array(list(names) + BLANK_NAMES), bv=bv, bq=bq,
               d=idx["d"], t=idx["t"], l=idx["l"], reg=idx["reg"])
    np.savez(path, **out)
    return out
