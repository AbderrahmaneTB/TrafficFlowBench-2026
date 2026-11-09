import numpy as np
from common import PANELS
from t4_odme import load_problem
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for P in PANELS:
    pr = {s: load_problem(P, s) for s in ("train", "validation", "private")}
    M = pr["train"]["M"]
    out = []
    for a, b in (("train", "validation"), ("train", "private"), ("validation", "private")):
        dc = np.log(pr[b]["c"] / pr[a]["c"])
        db = np.log((M @ pr[b]["b"]) / (M @ pr[a]["b"]))
        out.append(f"{a[:2]}-{b[:2]} corr(dlog c, dlog Ab)={np.corrcoef(dc, db)[0,1]:+.3f} sd dc={dc.std():.3f} sd db={db.std():.3f}")
    # does the prior's segment-load shape match the counts' shape within a split?
    s = []
    for sp in pr:
        ab = M @ pr[sp]["b"]; c = pr[sp]["c"]
        s.append(np.corrcoef(np.log(ab), np.log(c))[0, 1])
    # zone-level structure of the prior error: ratio Ab/c along the corridor
    r = (M @ pr["validation"]["b"]) / pr["validation"]["c"]
    print(P, " | ".join(out), "| corr(log Ab, log c) per split", np.round(s, 3))
    print("    Ab/c along corridor (val):", np.round(r[:: max(1, len(r) // 12)], 2))
