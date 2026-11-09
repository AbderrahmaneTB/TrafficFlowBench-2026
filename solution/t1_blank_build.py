"""Build the emulated-blank rows (t1_blank.blank_rows) for a list of panels / splits / groups.

  TFB_T1FEAT=7 python t1_blank_build.py --splits validation private --groups 0 1 2 --ngroups 3 [--panels ...]
"""
from __future__ import annotations

import argparse
import time

from common import QUEUE_PANELS
from t1_blank import blank_rows, blank_extra, blank_extra2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", default=QUEUE_PANELS)
    ap.add_argument("--splits", nargs="+", default=["validation", "private"])
    ap.add_argument("--groups", nargs="+", type=int, default=[0])
    ap.add_argument("--ngroups", type=int, default=3)
    ap.add_argument("--extras", action="store_true", help="also cache the extra / extra2 columns of every group")
    a = ap.parse_args()
    for g in a.groups:                         # every panel's first group before any second one
        for s in a.splits:
            for p in a.panels:
                t0 = time.time()
                r = blank_rows(p, s, g, a.ngroups)
                if a.extras:
                    blank_extra(p, s, g, a.ngroups); blank_extra2(p, s, g, a.ngroups)
                print(p, s, g, "rows", len(r["tv"]), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
