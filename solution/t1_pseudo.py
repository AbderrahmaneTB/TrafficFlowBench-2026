"""Self-supervised Task 1 rows from a scored month (validation or private).

Task 1 is offline, so every released observation of a split may be used. The scored months run on a
new scenario seed, so the most relevant training data are their own observed cells: hide a share
`rho` of the observed eligible cells (the official mask is iid per cell), rebuild the features from
the doubly-masked layer and learn to put those cells back. The real targets stay blank throughout.
"""
from __future__ import annotations

import os
import zlib

import numpy as np

from common import WORK, load
from t1_features import build, FEAT_VERSION

FEAT = f"{WORK}/t1feat"


def pseudo_rows(panel: str, split: str, rho: float = 0.15, n_max: int = 300_000, seed: int = 0) -> dict:
    """seed picks the hidden share; seed 0 is the original mask (same cache file as before seeds existed)."""
    path = f"{FEAT}/{panel}_{split}_pseudo_{int(rho * 100)}_{n_max}_f{FEAT_VERSION}{f'_s{seed}' if seed else ''}.npz"
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    z = load(panel, split)
    rng = np.random.default_rng(seed + zlib.crc32(f"{panel}|{split}".encode()))
    obs = z["melig"].astype(bool) & np.isfinite(z["mv"]) & np.isfinite(z["mq"])
    pick = obs & (rng.random(obs.shape) < rho)
    d, t, l = np.nonzero(pick)
    if len(d) > n_max:
        keep = np.sort(rng.choice(len(d), n_max, replace=False))
        d, t, l = d[keep], t[keep], l[keep]
    zz = dict(z)
    for k in ("mv", "mq", "mo"):
        a = z[k].copy(); a[pick] = np.nan; zz[k] = a        # hide every picked cell, not only the kept rows
    zz["v"], zz["q"] = z["mv"], z["mq"]                     # truth for build(): the observed values
    idx = dict(d=d.astype(np.int16), t=t.astype(np.int16), l=l.astype(np.int16), reg=z["regime"][d].astype(np.int8))
    X, names, bv, bq, ix, (tv, tq) = build(panel, split, zz, idx=idx)
    out = dict(X=X, names=np.array(names), bv=bv, bq=bq, tv=tv.astype(np.float32), tq=tq.astype(np.float32),
               reg=idx["reg"], d=idx["d"])
    np.savez(path, **out)
    return out
