"""Task 4: path-flow estimation on the linear corridor OD structure.

Every panel is a line of zones Z0..Z{n-1}; there is one path per OD pair o<d and the
released counts are constant within each zone-to-zone segment, so the count
constraints collapse to one number per segment: c_k = sum_{o<=k<d} f_od.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from common import RELEASE, WORK, PANELS, SPLITS


def load_problem(panel: str, split: str) -> dict:
    cache = f"{WORK}/t4/prob_{panel}_{split}.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    prob = _load_problem(panel, split)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(prob, fh)
    return prob


def _load_problem(panel: str, split: str) -> dict:
    net = f"{RELEASE}/corridors/{panel}/network"
    paths = pd.read_csv(f"{net}/path_set.csv", dtype=str)
    inc = pd.read_csv(f"{net}/path_link_incidence.csv", dtype=str)
    o = paths.origin_zone.str[1:].astype(int).to_numpy()
    d = paths.destination_zone.str[1:].astype(int).to_numpy()
    nz = int(max(o.max(), d.max()) + 1)
    counts = pd.read_csv(f"{RELEASE}/task4/{panel}/{split}/synthetic_link_counts.csv", dtype={"link_id": str})
    prior = pd.read_csv(f"{RELEASE}/task4/{panel}/{split}/synthetic_weak_prior.csv", dtype={"path_id": str})
    prior = prior.set_index("path_id").reindex(paths.path_id)
    b = prior.path_flow.to_numpy(float)
    # segment of each counted link: the paths using it are exactly {o <= k < d}
    pid = {p: i for i, p in enumerate(paths.path_id)}
    seg_of_link = {}
    use = inc[inc.link_id.isin(set(counts.link_id))]
    for link, g in use.groupby("link_id"):
        pi = g.path_id.map(pid).to_numpy()
        lo, hi = o[pi].max(), d[pi].min() - 1
        assert lo == hi, (panel, link, lo, hi)
        k = int(lo)
        expected = np.flatnonzero((o <= k) & (d > k))
        assert len(expected) == len(pi), (panel, link)
        seg_of_link[link] = k
    counts["seg"] = counts.link_id.map(seg_of_link)
    seg_counts = counts.groupby("seg")["count"].agg(["mean", "std", "size"])
    assert (seg_counts["std"].fillna(0) < 1e-6).all()
    segs = seg_counts.index.to_numpy(int)
    c = seg_counts["mean"].to_numpy(float)
    # incidence restricted to segments: M[k, p] = 1 if o_p <= seg_k < d_p
    M = ((o[None, :] <= segs[:, None]) & (d[None, :] > segs[:, None])).astype(float)
    return dict(panel=panel, split=split, paths=paths, o=o, d=d, nz=nz, b=b, segs=segs, c=c, M=M,
                counts=counts, seg_size=seg_counts["size"].to_numpy(int),
                departure_time=str(prior.departure_time.iloc[0]))


def s_link(prob: dict, f: np.ndarray) -> float:
    """Exact local S_link: counted links share their segment's count."""
    loaded = prob["M"] @ f
    w = prob["seg_size"]
    return max(0.0, 1.0 - float(np.sum(w * np.abs(loaded - prob["c"]))) / float(np.sum(w * prob["c"])))


def nnls_l2(prob: dict, prior: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """min ||A f - c||^2 + lam ||f - prior||^2, f >= 0 (counted links weighted like the baseline)."""
    M, c, n = prob["M"], prob["c"], prob["seg_size"].astype(float)
    scale = float(np.mean(c))

    def fun(x):
        r = M @ x - c
        obj = np.sum(n * r * r) + lam * np.sum((x - prior) ** 2)
        grad = 2 * (M.T @ (n * r)) + 2 * lam * (x - prior)
        return obj / scale ** 2, grad / scale ** 2

    res = minimize(fun, np.maximum(prior, 0), jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * len(prior),
                   options=dict(maxiter=20000, maxfun=40000, ftol=1e-16, gtol=1e-12, maxcor=30))
    return res.x


def nnls_weighted(prob: dict, prior: np.ndarray, w: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """min ||A f - c||^2 + lam * sum w_p (f_p - prior_p)^2, f >= 0. w = 1 is nnls_l2; w ~ 1/prior moves
    towards the multiplicative (KL-like) correction, w ~ prior away from it."""
    M, c, n = prob["M"], prob["c"], prob["seg_size"].astype(float)
    scale = float(np.mean(c))

    def fun(x):
        r = M @ x - c
        obj = np.sum(n * r * r) + lam * np.sum(w * (x - prior) ** 2)
        grad = 2 * (M.T @ (n * r)) + 2 * lam * w * (x - prior)
        return obj / scale ** 2, grad / scale ** 2

    res = minimize(fun, np.maximum(prior, 0), jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * len(prior),
                   options=dict(maxiter=20000, maxfun=40000, ftol=1e-16, gtol=1e-12, maxcor=30))
    return res.x


def path_km(panel: str, paths: pd.DataFrame) -> np.ndarray:
    links = pd.read_csv(f"{RELEASE}/corridors/{panel}/network/links.csv", dtype={"link_id": str}).set_index("link_id").length_km
    return np.array([sum(links.get(x, 0.0) for x in s.split(";")) for s in paths.link_seq])


def gravity_fit(prob: dict, logb: np.ndarray, n_bins: int = 25) -> np.ndarray:
    """Least-squares fit log b_od = a_o + c_d + h(trip length), h piecewise constant in km bins."""
    o, d, nz = prob["o"], prob["d"], prob["nz"]
    km = prob.get("km")
    if km is None:
        km = path_km(prob["panel"], prob["paths"]); prob["km"] = km
    edges = np.quantile(km, np.linspace(0, 1, n_bins + 1)[1:-1])
    kb = np.searchsorted(edges, km)
    n = len(o)
    rows = np.arange(n)
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
    X = csr_matrix((np.ones(3 * n), (np.r_[rows, rows, rows], np.r_[o, nz + d, 2 * nz + kb])),
                   shape=(n, 2 * nz + n_bins))
    sol = lsqr(X, logb, damp=1e-6, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    return X @ sol


def kl_projection(prob: dict, prior: np.ndarray, iters: int = 100) -> np.ndarray:
    """argmin KL(f || prior) s.t. M f = c  ->  f = prior * exp(M^T mu), Newton on mu."""
    M, c = prob["M"], prob["c"]
    mu = np.zeros(len(c))
    for _ in range(iters):
        f = prior * np.exp(M.T @ mu)
        g = M @ f - c
        if np.max(np.abs(g) / c) < 1e-10:
            break
        H = (M * f[None, :]) @ M.T
        step = np.linalg.solve(H + 1e-9 * np.eye(len(c)), g)
        # damped Newton on the convex dual
        t = 1.0
        dual = lambda m: np.sum(prior * np.exp(M.T @ m)) - c @ m
        f0 = dual(mu)
        while dual(mu - t * step) > f0 - 1e-4 * t * (g @ step) and t > 1e-8:
            t *= 0.5
        mu = mu - t * step
    return prior * np.exp(M.T @ mu)


if __name__ == "__main__":
    for panel in PANELS:
        for split in SPLITS:
            prob = load_problem(panel, split)
            b = prob["b"]
            f1 = nnls_l2(prob, b)
            f2 = kl_projection(prob, b)
            print(f"{panel:11s} {split:10s} Z={prob['nz']:3d} segs={len(prob['c']):3d} "
                  f"S_link prior={s_link(prob, b):.4f} nnls={s_link(prob, f1):.5f} kl={s_link(prob, f2):.6f} "
                  f"sum b={b.sum():.0f} nnls={f1.sum():.0f} kl={f2.sum():.0f} zeros nnls={np.mean(f1 < 1e-6):.3f} "
                  f"L1(nnls,kl)/sum={np.abs(f1 - f2).sum() / f2.sum():.3f}", flush=True)
