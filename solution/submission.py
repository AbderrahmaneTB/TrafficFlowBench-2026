"""Map per-task predictions onto sample_submission.csv through submission_key.csv."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import RELEASE, WORK, PANELS, SPLITS, load

KEYIDX = f"{WORK}/key_index.npz"
# every released speed of every split is at most 120.75 km/h (2-18 % of a panel's cells sit exactly there): the
# generator caps speed, so a prediction above the cap is never closer to the truth than the cap itself
VMAX = 120.75


def key_index() -> dict:
    if os.path.exists(KEYIDX):
        z = np.load(KEYIDX, allow_pickle=True)
        return {k: z[k] for k in z.files}
    lidx, pidx, didx, org = {}, {}, {}, {}
    for p in PANELS:
        lidx[p] = {x: i for i, x in enumerate(str(y) for y in np.load(f"{WORK}/cache/{p}_validation.npz")["links"])}
        pidx[p] = {x: i for i, x in enumerate(pd.read_csv(f"{RELEASE}/corridors/{p}/network/path_set.csv", dtype=str).path_id)}
        for s in SPLITS[1:]:
            didx[(p, s)] = {x: i for i, x in enumerate(str(y) for y in np.load(f"{WORK}/cache/{p}_{s}.npz")["dates"])}
            try:
                wi = pd.read_csv(f"{RELEASE}/task2/{p}/{s}/window_index.csv")
                org.update(dict(zip(wi.window_id, pd.to_datetime(wi.forecast_origin, utc=True))))
            except FileNotFoundError:
                pass
    cols = dict(task=[], split=[], panel=[], d=[], t=[], l=[], h=[], path=[], wid=[], sid=[])
    reader = pd.read_csv(f"{RELEASE}/submission_key.csv", dtype=str, chunksize=500_000,
                         usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id", "window_id", "path_id"])
    for k in reader:
        n = len(k)
        task = k.task.map({"state": 0, "queue": 1, "odme": 2}).to_numpy(np.int8)
        split = k.split.map({s: i for i, s in enumerate(SPLITS)}).to_numpy(np.int8)
        panel = k.panel.map({p: i for i, p in enumerate(PANELS)}).to_numpy(np.int8)
        d = np.full(n, -1, np.int16); t = np.full(n, -1, np.int16); l = np.full(n, -1, np.int16)
        h = np.full(n, -1, np.int8); path = np.full(n, -1, np.int32); wid = np.full(n, "", object)
        for (pi, si), g in pd.DataFrame({"pi": panel, "si": split}).groupby(["pi", "si"]).groups.items():
            p, s = PANELS[pi], SPLITS[si]
            g = np.asarray(g)
            sub = k.iloc[g]; tk = task[g]
            ms = g[tk == 0]
            if len(ms):
                ts = k.timestamp.iloc[ms]
                d[ms] = ts.str[:10].map(didx[(p, s)]).to_numpy()
                t[ms] = (ts.str[11:13].astype(int) * 12 + ts.str[14:16].astype(int) // 5).to_numpy()
                l[ms] = k.link_id.iloc[ms].map(lidx[p]).to_numpy()
            mq = g[tk == 1]
            if len(mq):
                tsq = pd.to_datetime(k.timestamp.iloc[mq], utc=True)
                o = pd.to_datetime(k.window_id.iloc[mq].map(org), utc=True)
                h[mq] = ((tsq - o).dt.total_seconds() // 300).astype(int).to_numpy()
                l[mq] = k.link_id.iloc[mq].map(lidx[p]).to_numpy()
                wid[mq] = k.window_id.iloc[mq].to_numpy()
            mo = g[tk == 2]
            if len(mo):
                path[mo] = k.path_id.iloc[mo].map(pidx[p]).to_numpy()
        for name, arr in (("task", task), ("split", split), ("panel", panel), ("d", d), ("t", t), ("l", l),
                          ("h", h), ("path", path), ("wid", wid)):
            cols[name].append(arr)
        cols["sid"].append(k.submission_id.astype(np.int64).to_numpy())
    out = {k: np.concatenate(v) for k, v in cols.items()}
    sid = out.pop("sid")
    assert sid[0] == 1 and (np.diff(sid) == 1).all()
    task = out["task"]
    assert (out["d"][task == 0] >= 0).all() and (out["l"][task != 2] >= 0).all() and (out["path"][task == 2] >= 0).all()
    assert ((out["h"][task == 1] >= 1) & (out["h"][task == 1] <= 6)).all()
    out["wid"] = out["wid"].astype(str)
    np.savez(KEYIDX, **out)
    return out


def assemble(state: dict, queue: dict, odme: dict, out_path: str) -> pd.DataFrame:
    """state[(panel, split)] = (speed[D,288,L], flow[D,288,L]); queue[window_id] = int8[6, L];
    odme[(panel, split)] = path_flow[P] in path_set order."""
    K = key_index()
    n = len(K["task"])
    sp = np.zeros(n); fl = np.zeros(n); qp = np.zeros(n, np.int8); pf = np.zeros(n)
    for pi, p in enumerate(PANELS):
        for si, s in enumerate(SPLITS[1:], start=1):
            m = (K["task"] == 0) & (K["panel"] == pi) & (K["split"] == si)
            V, Q = state[(p, s)]
            sp[m] = V[K["d"][m], K["t"][m], K["l"][m]]
            fl[m] = Q[K["d"][m], K["t"][m], K["l"][m]]
            m = (K["task"] == 2) & (K["panel"] == pi) & (K["split"] == si)
            pf[m] = odme[(p, s)][K["path"][m]]
    mq = np.flatnonzero(K["task"] == 1)
    wids = K["wid"][mq]
    for w in np.unique(wids):
        r = mq[wids == w]
        qp[r] = queue[w][K["h"][r] - 1, K["l"][r]]
    st = K["task"] == 0
    assert np.isfinite(sp[st]).all() and np.isfinite(fl[st]).all(), "missing state values"
    assert (sp[st] > 0).all() and (fl[st] >= 0).all()
    assert np.isfinite(pf).all() and (pf >= 0).all()
    assert set(np.unique(qp)) <= {0, 1}
    assert float((fl[st] < 50).mean()) <= 0.2
    tpl = pd.read_csv(f"{RELEASE}/sample_submission.csv", usecols=["submission_id", "task"])
    assert len(tpl) == n and (tpl.submission_id.to_numpy() == np.arange(1, n + 1)).all()
    sub = pd.DataFrame({"submission_id": np.arange(1, n + 1), "task": tpl.task.to_numpy(),
                        "speed_kmh": np.where(st, np.round(np.minimum(sp, VMAX), 4), 0.0),
                        "flow_vph": np.where(st, np.round(fl, 3), 0.0),
                        "queue_pred": qp.astype(int),
                        "path_flow": np.where(K["task"] == 2, pf, 0.0)})
    sub.to_csv(out_path, index=False, float_format="%.6g")
    return sub
