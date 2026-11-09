"""Held-out February S_state of LightGBM, CatBoost and their blends (same holdout rows)."""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor

from common import WORK, PANELS, FAMILY
from t1_train import panel_rows, state_score

WEIGHTS = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)   # weight on CatBoost

if __name__ == "__main__":
    lg = {t: lgb.Booster(model_file=f"{WORK}/t1_{t}_v1.txt") for t in ("speed", "flow")}
    cb = {}
    for t in ("speed", "flow"):
        cb[t] = CatBoostRegressor(); cb[t].load_model(f"{WORK}/t1_{t}_cb1.cbm")
    res = {}
    for p in PANELS:
        z = panel_rows(p, 300_000, 100_000)
        h = z["hold"] & np.isfinite(z["tv"]) & np.isfinite(z["tq"])
        X = z["X"][h]
        pv = {"lg": z["bv"][h] + lg["speed"].predict(X, num_threads=8),
              "cb": z["bv"][h] + cb["speed"].predict(X, thread_count=8)}
        pq = {"lg": z["bq"][h] + lg["flow"].predict(X, num_threads=8),
              "cb": z["bq"][h] + cb["flow"].predict(X, thread_count=8)}
        res[p] = {}
        for w in WEIGHTS:
            v = np.clip((1 - w) * pv["lg"] + w * pv["cb"], 3, 140)
            q = np.clip((1 - w) * pq["lg"] + w * pq["cb"], 0, 3500)
            res[p][w] = state_score(v, q, z["tv"][h], z["tq"][h], z["reg"][h])
        print(p, {w: round(s, 5) for w, s in res[p].items()}, flush=True)
    fam = sorted(set(FAMILY.values()))
    overall = {w: float(np.mean([np.mean([res[p][w] for p in PANELS if FAMILY[p] == f]) for f in fam])) for w in WEIGHTS}
    print("family mean:", {w: round(s, 5) for w, s in overall.items()})
    json.dump(dict(res={p: {str(w): s for w, s in r.items()} for p, r in res.items()}, overall=overall),
              open(f"{WORK}/t1_blend_eval.json", "w"), indent=1)
