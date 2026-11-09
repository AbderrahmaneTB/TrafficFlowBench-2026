"""Convert the released parquet files into dense per-panel numpy arrays.

Layout per panel and split (saved as one .npz in WORK/cache):
  dates  (D,)            date strings
  links  (L,)            link ids in milepost order
  v, q, o (D, 288, L)    speed, flow, occupancy (NaN where not released)
  pct    (D, 288, L)     pct_observed (uint8)
  elig   (D, 288, L)     is_score_eligible
  regime (D,)            0/1/2 for R1/R2/R3 (masked layer)
For train both the unmasked layer (v, q, o) and the masked layer (mv, mq, mo) are kept.
Ramp arrays: rq, rpct, relig (D, 288, R) with ramp ids/types/attachment.
"""
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import RELEASE as R, CACHE as OUT
os.makedirs(OUT, exist_ok=True)
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
COLS = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]
RCOLS = ["timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"]


def slot_of(ts):
    # 'YYYY-MM-DDTHH:MM:SSZ'
    h = ts.str.slice(11, 13).astype(int)
    m = ts.str.slice(14, 16).astype(int)
    return (h * 12 + m // 5).to_numpy()


def read_day(path, link_index, cols=COLS, key="link_id"):
    t = pq.ParquetFile(path).read(columns=cols).to_pandas()
    li = t[key].map(link_index).to_numpy()
    sl = slot_of(t.timestamp)
    return t, li, sl


def build(panel, splits=("train", "validation", "private")):
    pdir = f"{R}/corridors/{panel}"
    fd = pd.read_csv(f"{pdir}/network/fd_parameters.csv", dtype={"link_id": str})
    first = sorted(glob.glob(f"{pdir}/train/mainline_states/*/*.parquet"))[0]
    mp = pd.read_parquet(first, columns=["link_id", "milepost"]).drop_duplicates("link_id")
    mp = mp.sort_values(["milepost", "link_id"])
    links = mp.link_id.astype(str).tolist()
    link_index = {l: i for i, l in enumerate(links)}
    L = len(links)
    amap = pd.read_csv(f"{pdir}/network/ramp_attachment_map.csv", dtype=str)
    ramps = amap.ramp_link_id.tolist()
    ramp_index = {r: i for i, r in enumerate(ramps)}
    Rn = len(ramps)
    for split in splits:
        masked = sorted(glob.glob(f"{pdir}/{split}/mainline_states_masked/*/*.parquet"))
        by_date = {}
        for p in masked:
            date = os.path.basename(p)[len("synthetic_mainline_"):-len(".parquet")].replace("_", "-")
            reg = int(os.path.basename(os.path.dirname(p))[-1]) - 1
            by_date[date] = (p, reg)
        dates = sorted(by_date)
        D = len(dates)
        shape = (D, 288, L)
        arrs = {}
        names = ["mv", "mq", "mo", "mpct", "melig"]
        if split == "train":
            names += ["v", "q", "o", "pct", "elig"]
        for n in names:
            arrs[n] = np.full(shape, np.nan, np.float32) if n[-1] in "vqo" else np.zeros(shape, np.uint8)
        regime = np.zeros(D, np.int8)
        rq = np.full((D, 288, Rn), np.nan, np.float32)
        rpct = np.zeros((D, 288, Rn), np.uint8)
        relig = np.zeros((D, 288, Rn), np.uint8)

        def work(di):
            date = dates[di]
            p, reg = by_date[date]
            out = {}
            out["m"] = read_day(p, link_index)
            if split == "train":
                ym = date[:7]
                up = f"{pdir}/train/mainline_states/year_month={ym}/synthetic_mainline_{date.replace('-', '_')}.parquet"
                out["u"] = read_day(up, link_index)
            rp = glob.glob(f"{pdir}/{split}/ramp_states/*/*{date.replace('-', '_')}*.parquet")
            if rp:
                out["r"] = read_day(rp[0], ramp_index, RCOLS, "ramp_link_id")
            return di, reg, out

        with ThreadPoolExecutor(int(os.environ.get("TFB_IO_THREADS", "6"))) as ex:
            for di, reg, out in ex.map(work, range(D)):
                regime[di] = reg
                for pre, key in (("m", "m"), ("", "u")):
                    if key not in out:
                        continue
                    t, li, sl = out[key]
                    arrs[pre + "v"][di, sl, li] = t.speed_kmh.to_numpy(np.float32)
                    arrs[pre + "q"][di, sl, li] = t.flow_vph.to_numpy(np.float32)
                    arrs[pre + "o"][di, sl, li] = t.occupancy.to_numpy(np.float32)
                    arrs[pre + "pct"][di, sl, li] = t.pct_observed.to_numpy(np.uint8)
                    arrs[pre + "elig"][di, sl, li] = t.is_score_eligible.to_numpy(np.uint8)
                if "r" in out:
                    t, ri, sl = out["r"]
                    ok = ri >= 0
                    rq[di, sl[ok], ri[ok]] = t.flow_vph.to_numpy(np.float32)[ok]
                    rpct[di, sl[ok], ri[ok]] = t.pct_observed.to_numpy(np.uint8)[ok]
                    relig[di, sl[ok], ri[ok]] = t.is_score_eligible.to_numpy(np.uint8)[ok]
        np.savez(f"{OUT}/{panel}_{split}.npz", dates=np.array(dates), links=np.array(links),
                 ramps=np.array(ramps), ramp_type=amap.ramp_type.to_numpy(),
                 ramp_link=amap.nearest_mainline_link_id.to_numpy(), regime=regime,
                 rq=rq, rpct=rpct, relig=relig, **arrs)
        print(panel, split, D, "days", L, "links", Rn, "ramps", flush=True)


if __name__ == "__main__":
    # python build_cache.py [PANEL ...] [--splits validation private]   (default: every panel, all three splits)
    args = sys.argv[1:]
    splits = ("train", "validation", "private")
    if "--splits" in args:
        k = args.index("--splits")
        splits, args = tuple(args[k + 1:]), args[:k]
    for panel in (args or PANELS):
        build(panel, splits)
