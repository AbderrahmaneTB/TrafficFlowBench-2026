import numpy as np, pandas as pd
np.set_printoptions(precision=3, suppress=True, linewidth=220)
W = "C:/Users/HP/tfb/work/cache"; R = "C:/Users/HP/tfb/data/kaggle_public"
P = "D7_I405_N"
z = np.load(f"{W}/{P}_train.npz")
links = list(z["links"]); q = z["q"]; el = z["elig"].astype(bool)
c = pd.read_csv(f"{R}/task4/{P}/train/synthetic_link_counts.csv", dtype={"link_id": str}).set_index("link_id")["count"]
qq = np.where(el, q, np.nan)
for (h0, h1) in ((15, 19), (16, 18), (0, 24), (6, 10)):
    m = np.nanmean(qq[:, h0 * 12:h1 * 12, :], axis=(0, 1))
    s = pd.Series(m, index=links).reindex(c.index)
    r = (c / s)
    print(f"hours {h0}-{h1}: count/meanflow ratio median {r.median():.4f} sd {r.std():.4f} corr {np.corrcoef(c, s)[0,1]:.4f}")
# along corridor
m = pd.Series(np.nanmean(qq[:, 15 * 12:19 * 12, :], axis=(0, 1)), index=links)
mp = pd.read_parquet(sorted(__import__('glob').glob(f"{R}/corridors/{P}/train/mainline_states/*/*.parquet"))[0], columns=["link_id", "milepost"]).drop_duplicates("link_id").set_index("link_id").milepost
df = pd.DataFrame({"count": c, "pm_flow": m.reindex(c.index), "mp": mp.reindex(c.index)}).sort_values("mp")
print(df.iloc[::10].round(1).to_string())
# segment structure: how many distinct counts and where they change
print("distinct counts", df["count"].round(6).nunique())
