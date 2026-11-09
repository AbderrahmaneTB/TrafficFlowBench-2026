"""Upload a built submission and log it next to its build metadata."""
from __future__ import annotations

import json
import os
import sys
import time
import zipfile

from common import WORK
from kaggle_upload import submit

COMP = "2026-ieee-big-data-traffic-flow-bench"

if __name__ == "__main__":
    name, message = sys.argv[1], sys.argv[2]
    path = f"{WORK}/subs/{name}.csv"
    meta = json.load(open(f"{WORK}/subs/{name}.json"))
    msg = f"{name} {message} sha={meta['sha']}"
    # the raw CSV (~240 MB) drops the connection; a zip is ~4x smaller
    zpath = f"{WORK}/subs/{name}.zip"
    if not os.path.exists(zpath):
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(path, arcname=f"{name}.csv")
    t0 = time.time()
    ok, out = True, ""
    try:
        out = submit(zpath, msg, COMP)        # chunked upload, fresh connection for the submission itself
    except Exception as e:
        ok, out = False, repr(e)
    print(out, flush=True)
    with open(f"{WORK}/subs/submitted.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(name=name, message=msg, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                upload_s=round(time.time() - t0), ok=ok, result=out[:300], meta=meta)) + "\n")
