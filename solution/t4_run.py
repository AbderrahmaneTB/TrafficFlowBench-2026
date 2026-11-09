"""Write Task 4 path-flow variants for validation and private."""
from __future__ import annotations

import os
import sys

import numpy as np

from common import WORK, PANELS
from t4_odme import load_problem, kl_projection, nnls_l2, nnls_weighted, s_link, gravity_fit

OUT = f"{WORK}/t4"
os.makedirs(OUT, exist_ok=True)


def variant(name: str, prob: dict, others: list[dict]) -> np.ndarray:
    b = prob["b"]
    if name == "nnls":
        return nnls_l2(prob, b, 0.05)
    if name == "kl":
        return kl_projection(prob, b)
    if name.startswith("klblend"):
        # geometric blend of the split's own prior with the other released priors
        w_own = float(name[len("klblend"):] or 50) / 100
        lb = w_own * np.log(b) + (1 - w_own) * np.mean([np.log(o["b"]) for o in others], axis=0)
        return kl_projection(prob, np.exp(lb))
    if name.startswith("klgrav"):
        # shrink each path's log prior toward a gravity fit: weight on the raw deviation
        lam = float(name[len("klgrav"):] or 0) / 100
        lb = np.log(b)
        g = gravity_fit(prob, lb)
        return kl_projection(prob, np.exp(g + lam * (lb - g)))
    if name.startswith("nnlspw"):
        # weighted L2: w_p = (b_p / mean b)^(-gamma); gamma 0 = nnls, gamma 1 ~ chi-square (KL-like); "m" = minus
        g = float(name[len("nnlspw"):].replace("m", "-")) / 100
        return nnls_weighted(prob, b, (b / b.mean()) ** (-g), 0.05)
    if name.startswith("nnlsw"):
        # log prior = w * own + (1-w) * mean of the other two splits. Count changes between splits track
        # prior-load changes (corr ~0.75), so ~half of a prior's per-path variance is real split demand:
        # optimal w = (3a + e) / 3(a + e) ~ 0.7 for a share a ~ 0.55 of real variation
        w = float(name[len("nnlsw"):]) / 100
        lo = np.mean([np.log(o["b"]) for o in others], axis=0)
        return nnls_l2(prob, np.exp(w * np.log(b) + (1 - w) * lo), 0.05)
    if name.startswith("nnlseb"):
        # empirical-Bayes base: pooled gravity fit plus the shrunk per-path residual of the
        # geometric mean (structure variance / (structure + noise/3)); nnlseb0 = pure gravity
        L = np.stack([np.log(prob["b"])] + [np.log(o["b"]) for o in others])
        gm = L.mean(0)
        g = gravity_fit(prob, gm)
        noise = L.var(0, ddof=1).mean() / len(L)
        struct = max((gm - g).var() - noise, 0.0)
        w = struct / (struct + noise)
        if name[len("nnlseb"):]:
            w = float(name[len("nnlseb"):]) / 100
        print(f"   {prob['panel']} {prob['split']} eb weight {w:.3f}", flush=True)
        return nnls_l2(prob, np.exp(g + w * (gm - g)), 0.05)
    if name.startswith(("nnlsgm", "klgm")):
        # the three split priors are independent lognormal draws around one base
        # (corr of successive log differences = -0.5): their geometric mean has a third of the noise
        gm = np.exp(np.mean([np.log(prob["b"])] + [np.log(o["b"]) for o in others], axis=0))
        if name.endswith("s"):
            gm *= b.sum() / gm.sum()
        return nnls_l2(prob, gm, 0.05) if name.startswith("nnls") else kl_projection(prob, gm)
    if name.startswith("kltemp"):
        alpha = float(name[len("kltemp"):]) / 100
        return kl_projection(prob, np.exp(alpha * np.log(b)))
    raise ValueError(name)


if __name__ == "__main__":
    names = sys.argv[1:] or ["nnls", "kl", "klblend50"]
    for name in names:
        res = {}
        for p in PANELS:
            probs = {s: load_problem(p, s) for s in ("train", "validation", "private")}
            for s in ("validation", "private"):
                if name.startswith(("nnlsgm", "klgm", "nnlseb", "nnlsw")):
                    others = [probs[x] for x in ("train", "validation", "private") if x != s]
                else:
                    others = [probs["train"]]          # train is released in full for every task
                f = variant(name, probs[s], others)
                res[f"{p}|{s}"] = f
                print(name, p, s, f"S_link={s_link(probs[s], f):.6f} sum={f.sum():.0f}", flush=True)
        np.savez(f"{OUT}/{name}.npz", **res)
