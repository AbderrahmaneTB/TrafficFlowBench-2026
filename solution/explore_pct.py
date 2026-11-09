import numpy as np, pandas as pd, sys
sys.path.insert(0, "c:/Users/HP/OneDrive/vs_code/kaglle_iee3500/solution")
from common import load, network
P = "D7_I405_N"
z = load(P, "train"); net = network(P)
v, q, o, pct, el = z["v"], z["q"], z["o"], z["pct"], z["elig"].astype(bool)
vf = net.free_speed_kmh.to_numpy()
# robust per-link free speed from eligible free-flow cells
ff = el & np.isfinite(v) & (v > 0.85 * vf)
vfl = np.nanmedian(np.where(ff, v, np.nan).reshape(-1, v.shape[2]), axis=0)
r = v - vfl
# smooth flow reference: mean of the 2 neighbouring links at same time (eligible)
qn = np.full_like(q, np.nan)
qe = np.where(el, q, np.nan)
qn[:, :, 1:-1] = np.nanmean(np.stack([qe[:, :, :-2], qe[:, :, 2:]]), axis=0)
for lev in [20, 25, 33, 40, 50, 60, 67, 75, 80, 83, 100]:
    m = (pct == lev) & np.isfinite(v) & (v > 0.85 * vf)
    rel = (q[m] / qn[m])
    rel = rel[np.isfinite(rel)]
    print(f"pct={lev:3d} n={m.sum():8d} speed resid sd={np.nanstd(r[m]):.3f} mean={np.nanmean(r[m]):+.3f} | flow/neighbour mean={np.mean(rel):.4f} sd={np.std(rel):.4f}")
