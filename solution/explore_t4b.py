import numpy as np, pandas as pd
np.set_printoptions(precision=4, suppress=True, linewidth=200)
R = "C:/Users/HP/tfb/data/kaggle_public"
PANELS = ["D7_I10_E", "D7_I210_E", "D7_I405_N", "D12_I5_N", "D12_I405_S"]
for P in PANELS:
    net = f"{R}/corridors/{P}/network"
    paths = pd.read_csv(f"{net}/path_set.csv", dtype=str)
    o = paths.origin_zone.str[1:].astype(int).to_numpy(); d = paths.destination_zone.str[1:].astype(int).to_numpy()
    B = {}
    for s in ("train", "validation", "private"):
        b = pd.read_csv(f"{R}/task4/{P}/{s}/synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").reindex(paths.path_id)
        B[s] = b.path_flow.to_numpy()
    lb = {s: np.log(B[s]) for s in B}
    print(P, "corr log prior train/val %.3f train/priv %.3f val/priv %.3f" % (
        np.corrcoef(lb["train"], lb["validation"])[0, 1], np.corrcoef(lb["train"], lb["private"])[0, 1],
        np.corrcoef(lb["validation"], lb["private"])[0, 1]))
    # ratio distribution between splits
    r = lb["validation"] - lb["train"]
    print("   log(b_val/b_train) sd %.3f  percentiles %s" % (r.std(), np.percentile(r, [5, 25, 50, 75, 95])))
    print("   b percentiles", np.percentile(B["train"], [0, 1, 5, 25, 50, 75, 95, 99, 100]))
    # log-linear fit: log b = a_o + b_d + g(dist)
    Z = max(d.max(), o.max()) + 1
    dist = d - o
    X = np.zeros((len(o), 2 * Z + 1))
    X[np.arange(len(o)), o] = 1; X[np.arange(len(o)), Z + d] = 1; X[:, 2 * Z] = dist
    for s in ("train",):
        coef, *_ = np.linalg.lstsq(X, lb[s], rcond=None)
        res = lb[s] - X @ coef
        print(f"   {s}: gravity loglin resid sd {res.std():.3f}, dist coef {coef[-1]:.4f}")
        # residual correlation across splits
        coefv, *_ = np.linalg.lstsq(X, lb["validation"], rcond=None)
        resv = lb["validation"] - X @ coefv
        print("   resid corr train/val %.3f" % np.corrcoef(res, resv)[0, 1])
    # distance profile
    prof = pd.Series(lb["train"]).groupby(dist).mean()
    print("   mean log b by dist (first 10):", prof.iloc[:10].round(3).to_numpy(), " last:", prof.iloc[-5:].round(3).to_numpy())
    # is prior exactly some function? check distinct values
    print("   distinct prior values:", len(np.unique(np.round(B["train"], 8))), "of", len(B["train"]))
