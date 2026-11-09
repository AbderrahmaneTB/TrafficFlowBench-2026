"""Experiment: does a direct density model raise the Task 3 surrogate without hurting S_state?

Variants on February train days (CatBoost speed/flow as the start):
  cb      : CatBoost speed and flow as submitted today
  kq      : flow replaced by k_model * speed
  kmix    : flow = 0.5 * CatBoost flow + 0.5 * k_model * speed
  kgeo    : the density correction split between speed and flow (q*v kept, q/v = k_model)"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor

from common import WORK, load, network
from t1_features import build, target_index
from t3_surrogate import PhysicsFrame


def evaluate(panel, cb, km, month="2031-02"):
    z = load(panel, "train")
    dates = [str(x) for x in z["dates"]]
    idx = target_index(panel, "train", dates, [str(x) for x in z["links"]])
    feb = np.array([d.startswith(month) for d in dates])
    sel = np.flatnonzero(feb[idx["d"]])
    X, names, bv, bq, ix, (tv, tq) = build(panel, "train", z, sel)
    lanes = network(panel).lanes.to_numpy(np.float32)
    pv = np.clip(bv + cb["speed"].predict(X, thread_count=10), 3, 140)
    pq = np.clip(bq + cb["flow"].predict(X, thread_count=10), 0, 3500)
    kb = bq / np.maximum(bv, 1.0)
    k_m = np.maximum(kb + km.predict(np.column_stack([X, kb]), num_threads=10), 0.0)
    k_cb = pq / np.maximum(pv, 1.0)
    r = np.sqrt(np.maximum(k_m, 1e-6) / np.maximum(k_cb, 1e-6))
    variants = {"cb": (pv, pq), "kq": (pv, np.clip(k_m * pv, 0, 3500)),
                "kmix": (pv, np.clip(0.5 * pq + 0.5 * k_m * pv, 0, 3500)),
                "kgeo": (np.clip(pv / r, 3, 140), np.clip(pq * r, 0, 3500))}
    d, t, l = ix["d"], ix["t"], ix["l"]
    el = z["elig"].astype(bool) & np.isfinite(z["v"]) & np.isfinite(z["q"])
    pf = PhysicsFrame(panel, z)
    ok = np.isfinite(tv) & np.isfinite(tq)
    kt = tq / np.maximum(tv, 1.0)
    out = {}
    Ntrue = (np.where(el, z["q"], np.nan) / np.maximum(np.where(el, z["v"], np.nan), 1.0)) * pf.length
    for name, (vv, qq) in variants.items():
        Vs = np.where(el, z["v"], np.nan).copy(); Qs = np.where(el, z["q"], np.nan).copy()
        Vs[d, t, l] = vv; Qs[d, t, l] = qq * lanes[l]
        tg = np.zeros(el.shape, bool); tg[d, t, l] = True
        Nsub = (Qs / np.maximum(Vs, 1.0)) * pf.length
        regs = []
        for rg in range(3):
            days = [i for i in np.flatnonzero(feb) if z["regime"][i] == rg]
            if not days:
                continue
            m = ok & (z["regime"][d] == rg)
            rv = np.sqrt(np.mean((vv[m] - tv[m]) ** 2)); rq = np.sqrt(np.mean((qq[m] - tq[m]) ** 2))
            s_state = 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)
            s_fd, _ = pf.score(Vs, Qs, el, tg, days)
            num, den = pf.lwr(Nsub[days], Ntrue[days], el, days)
            regs.append(dict(s_state=s_state, s_fd=s_fd, s_lwr=max(0.0, 1 - num / den)))
        res = {k: float(np.mean([x[k] for x in regs])) for k in ("s_state", "s_fd", "s_lwr")}
        res["s_phys"] = (res["s_fd"] + 2 * res["s_lwr"]) / 3
        res["total_part"] = 0.35 * res["s_state"] + 0.15 * res["s_phys"]
        res["k_l1"] = float(np.mean(np.abs((qq / np.maximum(vv, 1.0) - kt)[ok])))
        out[name] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=["D12_I405_S", "D7_I210_W"])
    a = ap.parse_args()
    cb = {}
    for t in ("speed", "flow"):
        cb[t] = CatBoostRegressor(); cb[t].load_model(f"{WORK}/t1_{t}_cb1.cbm")
    km = lgb.Booster(model_file=f"{WORK}/t1_k_v1.txt")
    for p in a.panels:
        res = evaluate(p, cb, km)
        print(p, json.dumps({k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in res.items()}), flush=True)
