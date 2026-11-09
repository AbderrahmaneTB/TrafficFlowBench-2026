"""Leave-one-day-out location priors for Task 2 training windows (data "<d>L").

prof_qu (queued share by link and slot over every train day) and head_rate (share of the fitted onset
windows queued at T+30) both include the training window's own day. For a link that queued on that day
only, the prior is non-zero because of the target itself, so the model learns to trust small priors that
at inference come from other days (the public Task 2 notebook saw +0.008 LB from the same fix). Every
training window gets its priors without its own day; February windows too, so the held-out score is
honest. Validation/private windows keep the full train-month priors (their days are not in train).

  python t2_lodo.py "" 9 11        # writes train{d}L_*, copies validation/private{d} to {d}L
"""
from __future__ import annotations

import shutil
import sys

import numpy as np

from common import WORK, QUEUE_PANELS, load, network
from t2_build import flat
from t2_underlying import underlying_queue

OUT = f"{WORK}/t2"


def lodo(panel: str, data: str):
    z = dict(np.load(f"{OUT}/train{data}_{panel}.npz", allow_pickle=True))
    names = list(z["names"])
    zz = load(panel, "train")
    vcut = network(panel).vcut.to_numpy(np.float32)
    L = zz["v"].shape[2]
    v = np.where(zz["elig"].astype(bool), zz["v"], np.nan)          # as in t1_features.profiles
    fin = np.isfinite(v)
    ind = fin & (v <= vcut)
    S, N = ind.sum(0).astype(np.float64), fin.sum(0).astype(np.float64)
    vf, elf = flat(zz["v"]), flat(zz["elig"]).astype(bool)
    # head_rate labels: data "" counted the observed queue, data 3/9/11 the underlying state on eligible cells
    lab = (np.isfinite(vf) & (vf <= vcut) & elf) if data == "" else (underlying_queue(vf, vcut) & elf)
    cnt, n, by_day = np.zeros(L), 0, {}
    for i, c, h, d in zip(z["meta_i"], z["meta_cond"], z["meta_hold"], z["meta_date"]):
        if c == "queue_onset" and not h:
            cnt += lab[int(i) + 6]; n += 1
            a = by_day.setdefault(str(d), [np.zeros(L), 0]); a[0] += lab[int(i) + 6]; a[1] += 1
    hi = names.index("head_rate")
    pc = {k: names.index(f"prof_qu{k}") for k in (0, 3, 6)}
    X = z["X"]
    w0 = int(np.argmax(~z["meta_hold"]))
    s0 = min(int(z["meta_i"][w0]) % 288 + 6, 287)
    print(panel, repr(data), "check head", round(float(np.abs(cnt / max(n, 1) - z["head"]).max()), 5),
          "prof_qu6", round(float(np.nanmax(np.abs(X[w0][:, pc[6]] - S[s0] / np.maximum(N[s0], 1)))), 5), flush=True)
    for w, (i, d, hold) in enumerate(zip(z["meta_i"], z["meta_date"], z["meta_hold"])):
        day, slot = int(i) // 288, int(i) % 288
        for k, col in pc.items():
            s = min(slot + k, 287)
            den = N[s] - fin[day, s]
            X[w][:, col] = np.where(den > 0, (S[s] - ind[day, s]) / np.maximum(den, 1), np.nan)
        if not hold and str(d) in by_day:
            c_d, n_d = by_day[str(d)]
            X[w][:, hi] = (cnt - c_d) / max(n - n_d, 1)
    z["X"] = X
    np.savez(f"{OUT}/train{data}L_{panel}.npz", **z)
    for split in ("validation", "private"):
        shutil.copy(f"{OUT}/{split}{data}_{panel}.npz", f"{OUT}/{split}{data}L_{panel}.npz")


if __name__ == "__main__":
    for data in sys.argv[1:]:
        for p in QUEUE_PANELS:
            lodo(p, data)
