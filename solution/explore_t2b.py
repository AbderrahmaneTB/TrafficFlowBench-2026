import numpy as np, pandas as pd
np.set_printoptions(precision=2, suppress=True, linewidth=250)
W = "C:/Users/HP/tfb/work/cache"; R = "C:/Users/HP/tfb/data/kaggle_public"
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
rows = []
for P in PANELS:
    z = np.load(f"{W}/{P}_train.npz")
    links = list(z["links"]); dates = list(z["dates"]); v = z["v"]; el = z["elig"].astype(bool)
    mv = z["mv"]
    fd = pd.read_csv(f"{R}/corridors/{P}/network/fd_parameters.csv", dtype={"link_id": str}).set_index("link_id").reindex(links)
    vcut = 0.6 * fd.free_speed_kmh.to_numpy()
    wi = pd.read_csv(f"{R}/task2/{P}/train/window_index.csv")
    for w in wi[wi.condition == "queue_ongoing"].itertuples():
        di = dates.index(w.date); T = pd.Timestamp(w.forecast_origin); t = T.hour * 12 + T.minute // 5
        Q = (v[di] <= vcut) & el[di] & np.isfinite(v[di])
        E = el[di] & np.isfinite(v[di])
        ious_T = []; ious_Tm5 = []
        for h in range(1, 7):
            ok = E[t + h]
            a = Q[t] & ok; b = Q[t + h] & ok; c = Q[t - 1] & ok
            u = (a | b).sum(); ious_T.append((a & b).sum() / u if u else 1)
            u2 = (c | b).sum(); ious_Tm5.append((c & b).sum() / u2 if u2 else 1)
        # overall persistence IoU over 6 steps (state at T)
        okall = E[t + 1:t + 7]; pa = np.broadcast_to(Q[t], okall.shape) & okall; tb = Q[t + 1:t + 7] & okall
        iou_all = (pa & tb).sum() / max((pa | tb).sum(), 1)
        pb = np.broadcast_to(Q[t - 1], okall.shape) & okall
        iou_all5 = (pb & tb).sum() / max((pb | tb).sum(), 1)
        # was row T available in masked layer?
        avail_T = np.isfinite(mv[di, t]).mean()
        rows.append(dict(panel=P, w=w.window_id[-3:], T=w.forecast_origin[11:16], nQ_T=int(Q[t].sum()),
                         iou_steps_T=np.round(ious_T, 2).tolist(), persist_T=round(iou_all, 3), persist_Tm5=round(iou_all5, 3),
                         availT=round(avail_T, 2)))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df.to_string())
print("mean persistence IoU using T:", df.persist_T.mean(), " using T-5:", df.persist_Tm5.mean())
