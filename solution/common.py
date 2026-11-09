"""Shared paths, network parameters and cached arrays."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd

RELEASE = os.environ.get("TFB_DATA", "C:/Users/HP/tfb/data/kaggle_public")
WORK = os.environ.get("TFB_WORK", "C:/Users/HP/tfb/work")
CACHE = f"{WORK}/cache"
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
QUEUE_PANELS = PANELS[:8]
FAMILY = {p: p.rsplit("_", 1)[0] for p in PANELS}
SPLITS = ("train", "validation", "private")


def load(panel: str, split: str) -> dict:
    z = np.load(f"{CACHE}/{panel}_{split}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def network(panel: str) -> pd.DataFrame:
    """Per-link parameters in the cache's milepost order, with topology indices."""
    z = np.load(f"{CACHE}/{panel}_validation.npz")
    links = [str(x) for x in z["links"]]
    net = f"{RELEASE}/corridors/{panel}/network"
    fd = pd.read_csv(f"{net}/fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
    df = fd.reindex(links).reset_index().rename(columns={"index": "link_id"})
    df["vcut"] = 0.6 * df.free_speed_kmh
    topo = pd.read_csv(f"{net}/lwr_mainline_topology.csv", dtype=str).drop_duplicates("link_id").set_index("link_id")
    idx = {l: i for i, l in enumerate(links)}
    inc, out = [], []
    for l in links:
        if l in topo.index:
            inc.append([idx[x] for x in str(topo.loc[l, "incoming_link_ids"]).split(";") if x in idx])
            out.append([idx[x] for x in str(topo.loc[l, "outgoing_link_ids"]).split(";") if x in idx])
        else:
            inc.append([]); out.append([])
    df["inc"] = inc
    df["out"] = out
    first = sorted(__import__("glob").glob(f"{RELEASE}/corridors/{panel}/train/mainline_states/*/*.parquet"))[0]
    mp = pd.read_parquet(first, columns=["link_id", "milepost"]).drop_duplicates("link_id").set_index("link_id").milepost
    df["milepost"] = mp.reindex(links).to_numpy()
    # travel direction relative to milepost order (+1: traffic moves to higher milepost)
    s = [np.sign(df.milepost[j] - df.milepost[i]) for i in range(len(df)) for j in df.out[i]]
    df.attrs["direction"] = 1 if np.sum(s) >= 0 else -1
    return df


@lru_cache(maxsize=None)
def free_speed(panel: str) -> np.ndarray:
    """Per-link free-flow speed measured on train (median of free-flow observations)."""
    path = f"{WORK}/freespeed_{panel}.npy"
    if os.path.exists(path):
        return np.load(path)
    z = np.load(f"{CACHE}/{panel}_train.npz", allow_pickle=True)
    v = z["v"]; el = z["elig"].astype(bool)
    vf = network(panel).free_speed_kmh.to_numpy()
    ff = el & np.isfinite(v) & (v > 0.85 * vf)
    out = np.nanmedian(np.where(ff, v, np.nan).reshape(-1, v.shape[2]), axis=0).astype(np.float32)
    np.save(path, out)
    return out


def slot_strings(date: str) -> list[str]:
    return [f"{date}T{h:02d}:{m:02d}:00Z" for h in range(24) for m in range(0, 60, 5)]
