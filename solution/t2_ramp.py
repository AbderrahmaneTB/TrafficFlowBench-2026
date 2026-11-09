"""Task 2 data "12" = data "11" + ramp features at the origin T.

An on-ramp's flow collapses in the step its mainline link jams, so the attached ramps are a second, independent
sensor of the queue at T and of its edges (useful where the mainline cell at T is unobserved or ineligible), and an
off-ramp that empties marks spill-back. Only released ramp rows stamped at or before T are read: the 60-minute
window T-60 .. T of each window's origin, exactly like the mainline history.

Per mainline link (the sum of its attached ramps of each type, valid cells only = eligible, >= 75 % observed):
  r_on_T, r_off_T        flow at T (veh/h; NaN when an attached ramp is invalid or the link has none)
  r_on_rel, r_off_rel    (flow at T + 30) / (mean over T-60 .. T-5 + 30)
  r_on_min3              the lowest relative on-ramp flow over T-10 .. T
  r_on_dn3, r_on_up3     the lowest relative on-ramp flow over the links within 3 positions downstream / upstream
  r_low_share            share of the corridor's valid on-ramp links whose relative flow at T is below 0.3
  python t2_ramp.py   ->  work/t2/{train,validation,private}12_<panel>.npz, validationM12_<panel>.npz"""
from __future__ import annotations

import numpy as np

from common import WORK, QUEUE_PANELS, load

OUT = f"{WORK}/t2"
RAMP_NAMES = ["r_on_T", "r_off_T", "r_on_rel", "r_off_rel", "r_on_min3", "r_on_dn3", "r_on_up3", "r_low_share"]
HIST = 12
EPS = 30.0


def flat(a):
    return a.reshape(-1, a.shape[-1])


def ramp_matrices(z):
    """(N, L) on- and off-ramp flow per mainline link over the split's timeline; NaN where an attached ramp is invalid."""
    links = [str(x) for x in z["links"]]
    li = {l: i for i, l in enumerate(links)}
    rq = flat(z["rq"]).astype(np.float32)
    ok = flat(z["relig"]).astype(bool) & (flat(z["rpct"]) >= 75) & np.isfinite(rq)
    N, L = rq.shape[0], len(links)
    mats = {}
    for kind, types in (("on", ("OR", "ON")), ("off", ("FR", "OFF"))):
        m = np.zeros((N, L), np.float32); bad = np.zeros((N, L), bool); has = np.zeros(L, bool)
        for r, (t, lk) in enumerate(zip(z["ramp_type"], z["ramp_link"])):
            j = li.get(str(lk))
            if j is None or str(t).upper() not in types:
                continue
            m[:, j] += np.where(ok[:, r], rq[:, r], 0.0)
            bad[:, j] |= ~ok[:, r]
            has[j] = True
        m[bad] = np.nan
        m[:, ~has] = np.nan
        mats[kind] = m
    return mats["on"], mats["off"]


def window_features(on, off, i, order):
    """(L, 8) ramp features of the window whose origin is row i (rows i-12 .. i only)."""
    lo = max(i - HIST, 0)
    with np.errstate(all="ignore"):
        on_h, off_h = on[lo:i + 1], off[lo:i + 1]
        on_T, off_T = on_h[-1], off_h[-1]
        m_on = np.nanmean(on_h[:-1], axis=0) if len(on_h) > 1 else np.full(on.shape[1], np.nan)
        m_off = np.nanmean(off_h[:-1], axis=0) if len(off_h) > 1 else np.full(off.shape[1], np.nan)
        rel_on = (on_T + EPS) / (m_on + EPS)
        rel_off = (off_T + EPS) / (m_off + EPS)
        min3 = (np.nanmin(on_h[-3:], axis=0) + EPS) / (m_on + EPS)
        ro = rel_on[order]                                  # travel order, most upstream first
        L = len(ro)
        dn = np.full(L, np.nan); up = np.full(L, np.nan)
        for k in range(L):
            a = ro[k + 1:k + 4]; b = ro[max(k - 3, 0):k]
            if np.isfinite(a).any():
                dn[k] = np.nanmin(a)
            if np.isfinite(b).any():
                up[k] = np.nanmin(b)
        rank = np.empty(L, int); rank[order] = np.arange(L)
        dn, up = dn[rank], up[rank]
        fin = np.isfinite(rel_on)
        low = float(np.mean(rel_on[fin] < 0.3)) if fin.any() else np.nan
    return np.column_stack([on_T, off_T, rel_on, rel_off, min3, dn, up, np.full(len(on_T), low)]).astype(np.float32)


def origin_rows(z_cache, origins):
    dates = [str(d) for d in z_cache["dates"]]
    out = []
    for o in origins:
        o = str(o)
        d = dates.index(o[:10])
        hh, mm = int(o[11:13]), int(o[14:16])
        out.append(d * 288 + (hh * 60 + mm) // 5)
    return out


def extend(src, dst, on, off, rows):
    z = dict(np.load(src, allow_pickle=True))
    names = [str(n) for n in z["names"]]
    order = np.argsort(z["X"][0][:, names.index("link_pos")], kind="stable")
    R = np.stack([window_features(on, off, i, order) for i in rows]) if len(rows) else \
        np.zeros((0, z["X"].shape[1], len(RAMP_NAMES)), np.float32)
    z["X"] = np.concatenate([z["X"], R], axis=2)
    z["names"] = np.array(names + RAMP_NAMES)
    np.savez_compressed(dst, **z)
    fin = np.isfinite(R[..., 2])
    print(dst.split("/")[-1], z["X"].shape, f"links with a valid on-ramp at T {fin.mean():.2f}", flush=True)


def main():
    for p in QUEUE_PANELS:
        zt = load(p, "train")
        on, off = ramp_matrices(zt)
        tr = np.load(f"{OUT}/train11_{p}.npz", allow_pickle=True)
        extend(f"{OUT}/train11_{p}.npz", f"{OUT}/train12_{p}.npz", on, off, [int(i) for i in tr["meta_i"]])
        for s in ("validation", "private"):
            zs = load(p, s)
            on, off = ramp_matrices(zs)
            ev = np.load(f"{OUT}/{s}11_{p}.npz", allow_pickle=True)
            extend(f"{OUT}/{s}11_{p}.npz", f"{OUT}/{s}12_{p}.npz", on, off, origin_rows(zs, ev["origin"]))
            if s == "validation":
                em = np.load(f"{OUT}/validationM_{p}.npz", allow_pickle=True)
                extend(f"{OUT}/validationM_{p}.npz", f"{OUT}/validationM12_{p}.npz", on, off,
                       [int(i) for i in em["meta_i"]])


if __name__ == "__main__":
    main()
