import numpy as np, pandas as pd
np.set_printoptions(precision=4, suppress=True, linewidth=200)
W = "C:/Users/HP/tfb/work/cache"
R = "C:/Users/HP/tfb/data/kaggle_public"
P = "D7_I405_N"
z = np.load(f"{W}/{P}_train.npz")
links = z["links"]; v = z["v"]; q = z["q"]; o = z["o"]; pct = z["pct"]; el = z["elig"].astype(bool)
fd = pd.read_csv(f"{R}/corridors/{P}/network/fd_parameters.csv", dtype={"link_id": str}).set_index("link_id").reindex(links)
vf = fd.free_speed_kmh.to_numpy(np.float32); cap = fd.capacity_vph.to_numpy(np.float32); lanes = fd.lanes.to_numpy(np.float32)
D, T, L = v.shape
print("shape", v.shape)
print("pct levels", np.unique(pct, return_counts=True))
fin = np.isfinite(v)
for lev in np.unique(pct):
    m = pct == lev
    print(f"pct={lev}: frac={m.mean():.4f} finite={fin[m].mean():.3f} elig={el[m].mean():.3f}")
# free flow speed stats
ff = el & fin & (v > 0.85 * vf)
r = np.where(ff, v - vf, np.nan)
print("free-flow speed - vf: mean %.3f sd %.3f" % (np.nanmean(r), np.nanstd(r)))
lm = np.nanmean(r, axis=(0, 1)); ls = np.nanstd(r, axis=(0, 1))
print("per-link mean offset quantiles", np.nanpercentile(lm, [0, 5, 50, 95, 100]))
print("per-link sd quantiles", np.nanpercentile(ls, [0, 5, 50, 95, 100]))
# noise by pct level in free flow
for lev in (75, 100):
    m = ff & (pct == lev)
    print(f"free-flow pct={lev}: sd={np.nanstd(np.where(m, r, np.nan)):.3f} n={m.sum()}")
# temporal autocorrelation of free-flow speed residual
a = r[:, :-1, :]; b = r[:, 1:, :]; ok = np.isfinite(a) & np.isfinite(b)
print("lag1 autocorr of ff speed resid: %.4f" % np.corrcoef(a[ok], b[ok])[0, 1])
a2 = r[:, :-2, :]; b2 = r[:, 2:, :]; ok2 = np.isfinite(a2) & np.isfinite(b2)
print("lag2 autocorr: %.4f" % np.corrcoef(a2[ok2], b2[ok2])[0, 1])
# spatial correlation of ff residual between adjacent links (milepost order)
a = r[:, :, :-1]; b = r[:, :, 1:]; ok = np.isfinite(a) & np.isfinite(b)
print("adjacent-link corr of ff speed resid: %.4f" % np.corrcoef(a[ok], b[ok])[0, 1])
# Is the distribution of the residual gaussian? kurtosis
x = r[np.isfinite(r)]
print("ff resid percentiles", np.percentile(x, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]))
# occupancy vs density q/v
k = q / np.maximum(v, 1)
ok = el & fin & np.isfinite(o)
ratio = (o * 285.0)[ok] / k[ok]
print("o*285/(q/v) percentiles", np.percentile(ratio, [1, 5, 25, 50, 75, 95, 99]))
kl = (k / lanes)[ok]; ol = o[ok]
print("corr(o, k/lane)=%.4f" % np.corrcoef(ol, kl)[0, 1])
# regress occupancy on k per lane
A = np.vstack([kl, np.ones_like(kl)]).T
coef = np.linalg.lstsq(A, ol, rcond=None)[0]
print("o = %.6f * k_lane + %.6f" % tuple(coef), "resid sd", np.std(ol - A @ coef))
# flow temporal smoothness: second difference noise estimate in free flow
qq = np.where(el & fin, q, np.nan)
d1 = qq[:, 1:, :] - qq[:, :-1, :]
d2 = qq[:, 2:, :] - 2 * qq[:, 1:-1, :] + qq[:, :-2, :]
print("flow: sd(d1)=%.2f sd(d2)=%.2f  ratio %.3f (iid noise -> sqrt(6/2)=1.732)" % (np.nanstd(d1), np.nanstd(d2), np.nanstd(d2) / np.nanstd(d1)))
qr = q / np.maximum(qq, 1)
# relative flow noise vs mean
print("mean flow per lane", np.nanmean(qq / lanes))
# flow vs neighbor link flow same time
a = qq[:, :, :-1]; b = qq[:, :, 1:]; ok = np.isfinite(a) & np.isfinite(b)
rel = (b[ok] - a[ok]) / ((a[ok] + b[ok]) / 2)
print("adjacent link flow rel diff percentiles", np.percentile(rel, [1, 5, 25, 50, 75, 95, 99]), "sd", rel.std())
