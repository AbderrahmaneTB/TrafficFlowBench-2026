import numpy as np, pandas as pd
np.set_printoptions(precision=4, suppress=True, linewidth=200)
W = "C:/Users/HP/tfb/work/cache"
R = "C:/Users/HP/tfb/data/kaggle_public"
P = "D7_I405_N"
z = np.load(f"{W}/{P}_train.npz")
links = z["links"]; v = z["v"]; q = z["q"]; o = z["o"]; pct = z["pct"]; el = z["elig"].astype(bool)
fd = pd.read_csv(f"{R}/corridors/{P}/network/fd_parameters.csv", dtype={"link_id": str}).set_index("link_id").reindex(links)
vf = fd.free_speed_kmh.to_numpy(np.float32); lanes = fd.lanes.to_numpy(np.float32)
D, T, L = v.shape
fin = np.isfinite(v)
ff = el & fin & (v > 0.85 * vf)
r = np.where(ff, v - vf, np.nan)
# per (day, link) mean
dl = np.nanmean(r, axis=1)  # D x L
print("sd of per-(day,link) mean resid: %.3f" % np.nanstd(dl))
print("sd of per-link mean (over all days): %.3f" % np.nanstd(np.nanmean(r, axis=(0, 1))))
# remove per-link mean then per-(day,link)
r1 = r - np.nanmean(r, axis=(0, 1))[None, None, :]
dl1 = np.nanmean(r1, axis=1)
print("after removing link mean: sd per-(day,link) mean %.3f" % np.nanstd(dl1))
r2 = r1 - dl1[:, None, :]
for lag in (1, 2, 3, 6, 12, 36):
    a = r2[:, :-lag, :]; b = r2[:, lag:, :]; ok = np.isfinite(a) & np.isfinite(b)
    print(f"within-day resid lag{lag} corr {np.corrcoef(a[ok], b[ok])[0,1]:.4f}")
print("within-day resid sd %.3f" % np.nanstd(r2))
# time-of-day profile of the residual (is speed dependent on flow?)
tod = np.nanmean(r1, axis=(0, 2))
print("tod mean resid (every hour):", tod[::12])
# speed resid vs flow per lane in free flow
ql = q / lanes
bins = np.arange(0, 2600, 200)
idx = np.digitize(np.where(ff, ql, np.nan), bins)
for b_ in range(1, len(bins)):
    m = (idx == b_) & ff
    if m.sum() > 1000:
        print(f"q/lane {bins[b_-1]}-{bins[b_]}: mean v-vf {np.nanmean(r1[m]):.3f} sd {np.nanstd(r1[m]):.3f} n={m.sum()}")
# Look at a single link-day time series
d0, l0 = 40, 150
print("link", links[l0], "vf", vf[l0], "lanes", lanes[l0])
for t in range(84, 120):
    print(t, f"{v[d0,t,l0]:7.2f} {q[d0,t,l0]:8.1f} {o[d0,t,l0]*285:7.2f} {q[d0,t,l0]/max(v[d0,t,l0],1):7.2f} pct={pct[d0,t,l0]}")
