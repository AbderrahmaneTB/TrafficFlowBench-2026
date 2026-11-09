import numpy as np, pandas as pd, sys
np.set_printoptions(precision=1, suppress=True, linewidth=250)
W = "C:/Users/HP/tfb/work/cache"; R = "C:/Users/HP/tfb/data/kaggle_public"
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
summary = []
for P in PANELS:
    z = np.load(f"{W}/{P}_train.npz")
    links = list(z["links"]); dates = list(z["dates"]); v = z["v"]; el = z["elig"].astype(bool)
    fd = pd.read_csv(f"{R}/corridors/{P}/network/fd_parameters.csv", dtype={"link_id": str}).set_index("link_id").reindex(links)
    vcut = 0.6 * fd.free_speed_kmh.to_numpy()
    wi = pd.read_csv(f"{R}/task2/{P}/train/window_index.csv")
    hist = pd.read_parquet(f"{R}/task2/{P}/train/window_history.parquet")
    if P == PANELS[0]:
        print(hist.columns.tolist()); print(hist.head(2).T)
    for w in wi.itertuples():
        di = dates.index(w.date)
        T = pd.Timestamp(w.forecast_origin); t = T.hour * 12 + T.minute // 5
        h = hist[hist.window_id == w.window_id]
        hts = sorted(h.timestamp.unique())
        fut = v[di, t + 1:t + 7, :]; fel = el[di, t + 1:t + 7, :]
        qf = (fut <= vcut) & fel & np.isfinite(fut)
        past = v[di, t - 12:t + 1, :]; pel = el[di, t - 12:t + 1, :]
        qp = (past <= vcut) & pel & np.isfinite(past)
        steps = qf.sum(axis=1)
        qlinks = np.where(qf[-1])[0]
        summary.append(dict(panel=P, wid=w.window_id[-3:], cond=w.condition[6:], T=w.forecast_origin[11:16], hist_n_ts=len(hts),
                            hist_first=hts[0][11:16], hist_last=hts[-1][11:16], fut_q_per_step=steps.tolist(),
                            past_q_per_step=qp.sum(axis=1).tolist(), q_links_T30=qlinks.tolist()))
df = pd.DataFrame(summary)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80); pd.set_option("display.max_rows", 200)
print(df.to_string())
