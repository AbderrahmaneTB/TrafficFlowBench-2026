import pandas as pd, numpy as np, sys
R = "C:/Users/HP/tfb/data/kaggle_public"
P = "D7_I405_N"
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
f = pd.read_parquet(f"{R}/corridors/{P}/train/mainline_states/year_month=2030-06/synthetic_mainline_2030_06_01.parquet")
print(f.dtypes); print(f.shape); print(f.head(5).T)
print("n links", f.link_id.nunique(), "n stations", f.station_id.nunique(), "n ts", f.timestamp.nunique())
print(f.describe().T)
m = pd.read_parquet(f"{R}/corridors/{P}/train/mainline_states_masked/mask_regime=R1/synthetic_mainline_2030_06_01.parquet")
print("masked cols", list(m.columns)); print(m.shape)
r = pd.read_parquet(sorted(__import__('glob').glob(f"{R}/corridors/{P}/train/ramp_states/year_month=2030-06/*.parquet"))[0])
print(r.dtypes); print(r.head(3).T); print(r.shape, r.ramp_link_id.nunique())
print(r.describe().T)
