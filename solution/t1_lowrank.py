"""Low-rank completion of each month of the masked layer (Task 1 feature, offline).

The generator draws flow as (link scale) x (a day curve shared by the whole corridor) plus i.i.d. noise, and
speed as one free-flow level per link except inside jams. A small-rank fit of the month unfolded as
(day x slot, link) therefore estimates the noise-free value of a hidden cell from hundreds of observations,
where the local neighbours of a cell average only a few. Fitted by alternating ridge least squares on the
eligible observed cells of the masked layer (Task 1 targets are blank there), per panel, split and month.

Output per panel / split: WORK/t1lr/<panel>_<split>.npz with lr_q (flow per lane) and lr_v (speed), (D, 288, L).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from common import WORK, PANELS, load, network

OUT = f"{WORK}/t1lr"
RANK = 6
LAM = 5.0
ITERS = 12


def als(M: np.ndarray, W: np.ndarray, rank: int = RANK, lam: float = LAM, iters: int = ITERS, seed: int = 0):
    """M (N, L) with W (N, L) bool observed: returns the rank-r reconstruction around the per-link mean."""
    N, L = M.shape
    Wf = W.astype(np.float64)
    cnt = Wf.sum(0)
    mu = np.where(cnt > 0, (np.where(W, M, 0.0)).sum(0) / np.maximum(cnt, 1), 0.0)
    R = np.where(W, M - mu[None, :], 0.0)
    rng = np.random.default_rng(seed)
    V = rng.normal(scale=0.1, size=(L, rank))
    U = np.zeros((N, rank))
    eye = lam * np.eye(rank)
    for _ in range(iters):
        # per-row Gram matrices V^T diag(W_i) V as one matmul
        G = (Wf @ (V[:, :, None] * V[:, None, :]).reshape(L, -1)).reshape(N, rank, rank) + eye
        U = np.linalg.solve(G, (R @ V)[..., None])[..., 0]
        G = (Wf.T @ (U[:, :, None] * U[:, None, :]).reshape(N, -1)).reshape(L, rank, rank) + eye
        V = np.linalg.solve(G, (R.T @ U)[..., None])[..., 0]
    return U @ V.T + mu[None, :]


def panel_split(panel: str, split: str, force: bool = False):
    path = f"{OUT}/{panel}_{split}.npz"
    if os.path.exists(path) and not force:
        return path
    z = load(panel, split)
    lanes = network(panel).lanes.to_numpy(np.float32)
    dates = np.array([str(x) for x in z["dates"]])
    mon = np.array([d[:7] for d in dates])
    D, T, L = z["mv"].shape
    el = z["melig"].astype(bool) & np.isfinite(z["mv"]) & np.isfinite(z["mq"])
    out = {"lr_q": np.full((D, T, L), np.nan, np.float32), "lr_v": np.full((D, T, L), np.nan, np.float32)}
    for m in sorted(set(mon)):
        r = np.flatnonzero(mon == m)
        W = el[r].reshape(-1, L)
        for key, arr in (("lr_q", z["mq"][r] / lanes), ("lr_v", z["mv"][r])):
            M = np.nan_to_num(arr.reshape(-1, L).astype(np.float64))
            out[key][r] = als(M, W).reshape(len(r), T, L).astype(np.float32)
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return path


def lowrank(panel: str, split: str) -> dict:
    z = np.load(panel_split(panel, split))
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    splits = ("train", "validation", "private")
    for panel in (sys.argv[1:] or PANELS):
        for split in splits:
            t0 = time.time()
            panel_split(panel, split)
            print(panel, split, f"{time.time() - t0:.0f}s", flush=True)
