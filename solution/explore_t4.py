import numpy as np, pandas as pd
np.set_printoptions(precision=4, suppress=True, linewidth=200)
R = "C:/Users/HP/tfb/data/kaggle_public"
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
for P in PANELS:
    net = f"{R}/corridors/{P}/network"
    paths = pd.read_csv(f"{net}/path_set.csv", dtype=str)
    inc = pd.read_csv(f"{net}/path_link_incidence.csv", dtype=str)
    zones_o = paths.origin_zone.nunique(); zones_d = paths.destination_zone.nunique()
    allz = sorted(set(paths.origin_zone) | set(paths.destination_zone), key=lambda s: int(s[1:]))
    nod = paths.groupby(["origin_zone", "destination_zone"]).size()
    links = inc.link_id.unique()
    pid = {p: i for i, p in enumerate(paths.path_id)}; lid = {l: i for i, l in enumerate(links)}
    A = np.zeros((len(links), len(paths)))
    A[inc.link_id.map(lid), inc.path_id.map(pid)] = 1
    out = [P, f"paths={len(paths)} zones={len(allz)} o={zones_o} d={zones_d} paths/OD max={nod.max()} links_in_A={len(links)}"]
    print(" ".join(out))
    for s in ("train", "validation", "private"):
        c = pd.read_csv(f"{R}/task4/{P}/{s}/synthetic_link_counts.csv", dtype={"link_id": str})
        b = pd.read_csv(f"{R}/task4/{P}/{s}/synthetic_weak_prior.csv", dtype={"path_id": str}).set_index("path_id").reindex(paths.path_id)
        bv = b.path_flow.to_numpy()
        rows = [lid[l] for l in c.link_id if l in lid]
        cv = c.set_index("link_id").reindex([l for l in c.link_id if l in lid])["count"].to_numpy()
        Ab = A[rows] @ bv
        rk = np.linalg.matrix_rank(A[rows])
        print(f"  {s}: counts={len(c)} inA={len(rows)} rank={rk} sum_prior={bv.sum():.0f} prior>0={np.mean(bv>0):.3f} "
              f"mean(Ab/c)={np.mean(Ab/cv):.3f} sd(Ab/c)={np.std(Ab/cv):.3f} c range [{cv.min():.0f},{cv.max():.0f}] "
              f"unique counts={len(np.unique(np.round(cv,6)))}")
