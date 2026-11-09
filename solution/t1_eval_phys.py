"""Score Task 1 models on February train days: official S_state and the Task 3 surrogate."""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np

from common import WORK, PANELS, load, network
from t1_features import build, target_index
from t3_surrogate import PhysicsFrame


def _pred(m, X):
    return m.predict(X, num_threads=10) if isinstance(m, lgb.Booster) else m.predict(X, thread_count=10)


def evaluate_panel(panel, models, month="2031-02", variants=("interp",)):
    z = load(panel, "train")
    dates = [str(x) for x in z["dates"]]
    links = [str(x) for x in z["links"]]
    idx = target_index(panel, "train", dates, links)
    feb = np.array([d.startswith(month) for d in dates])
    sel = np.flatnonzero(feb[idx["d"]])
    X, names, bv, bq, ix, (tv, tq) = build(panel, "train", z, sel)
    lanes = network(panel).lanes.to_numpy(np.float32)
    preds = {}
    for name, mods in models.items():
        pv = bv + np.mean([_pred(m, X) for m in mods["speed"]], axis=0)
        pq = bq + np.mean([_pred(m, X) for m in mods["flow"]], axis=0)
        preds[name] = (np.clip(pv, 3, 140), np.clip(pq, 0, 3500))
    if "interp" in variants:
        preds["interp"] = (bv, bq)
    d, t, l = ix["d"], ix["t"], ix["l"]
    el = z["elig"].astype(bool) & np.isfinite(z["v"]) & np.isfinite(z["q"])
    pf = PhysicsFrame(panel, z)
    reg_of_day = z["regime"]
    out = {}
    ok = np.isfinite(tv) & np.isfinite(tq)
    for name, (pv, pq) in preds.items():
        res = {}
        Vs = np.where(el, z["v"], np.nan); Qs = np.where(el, z["q"], np.nan)
        Vs = Vs.copy(); Qs = Qs.copy()
        Vs[d, t, l] = pv; Qs[d, t, l] = pq * lanes[l]
        tg = np.zeros(el.shape, bool); tg[d, t, l] = True
        Ntrue = (np.where(el, z["q"], np.nan) / np.maximum(np.where(el, z["v"], np.nan), 1.0)) * pf.length
        Nsub = (Qs / np.maximum(Vs, 1.0)) * pf.length
        regs = []
        for r in range(3):
            days = [i for i in np.flatnonzero(feb) if reg_of_day[i] == r]
            if not days:
                continue
            m = ok & (reg_of_day[d] == r)
            rv = np.sqrt(np.mean((pv[m] - tv[m]) ** 2)); rq = np.sqrt(np.mean((pq[m] - tq[m]) ** 2))
            s_state = 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)
            s_fd, _ = pf.score(Vs, Qs, el, tg, days)
            num, den = pf.lwr(Nsub[days], Ntrue[days], el, days)
            s_lwr = max(0.0, 1 - num / den)
            regs.append(dict(regime=r, s_state=s_state, rmse_v=rv, rmse_q=rq, s_fd=s_fd, s_lwr=s_lwr,
                             s_phys=(s_fd + 2 * s_lwr) / 3))
        out[name] = {k: float(np.mean([x[k] for x in regs])) for k in ("s_state", "rmse_v", "rmse_q", "s_fd", "s_lwr", "s_phys")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v1"], help="LightGBM tags (t1_*_<tag>.txt) or CatBoost tags (t1_*_<tag>.cbm)")
    ap.add_argument("--panels", nargs="+", default=PANELS)
    a = ap.parse_args()
    import os
    from catboost import CatBoostRegressor

    def _load(t, m):
        if os.path.exists(f"{WORK}/t1_{t}_{m}.cbm"):
            r = CatBoostRegressor(); r.load_model(f"{WORK}/t1_{t}_{m}.cbm"); return r
        return lgb.Booster(model_file=f"{WORK}/t1_{t}_{m}.txt")
    models = {m: {t: [_load(t, m)] for t in ("speed", "flow")} for m in a.models}
    allres = {}
    for p in a.panels:
        allres[p] = evaluate_panel(p, models)
        print(p, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in allres[p].items()}), flush=True)
    for name in list(a.models) + ["interp"]:
        print(name, {k: round(float(np.mean([allres[p][name][k] for p in allres])), 5)
                     for k in ("s_state", "s_fd", "s_lwr", "s_phys")})
