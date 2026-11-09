"""Competition upload in short resumable chunks.

`kaggle competitions submit` sends the whole file in one PUT; on a slow line (~65 kB/s, a 60 MB zip takes 15+ min)
the connection is dropped at the end and the upload is lost. Here every 8 MiB chunk is its own request (resumable
upload protocol: Content-Range, 308 until the last chunk), an interrupted chunk is resumed from the byte count the
server reports, and the submission itself is created on a fresh connection, with retries.
"""
from __future__ import annotations

import json
import os
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (ApiCreateSubmissionRequest,
                                                                  ApiStartSubmissionUploadRequest)

CHUNK = 8 * 1024 * 1024          # a multiple of 256 KiB, as the protocol requires


def _received(url: str, size: int):
    """Bytes the server holds (None when the upload is already complete)."""
    r = requests.put(url, headers={"Content-Length": "0", "Content-Range": f"bytes */{size}"}, timeout=120)
    if r.status_code in (200, 201):
        return None
    if r.status_code == 308:
        rng = r.headers.get("Range")
        return int(rng.split("-")[1]) + 1 if rng else 0
    raise RuntimeError(f"status query returned {r.status_code}: {r.text[:300]}")


def upload(path: str, url: str):
    size = os.path.getsize(path)
    pos, t0 = 0, time.time()
    with open(path, "rb") as f:
        while True:
            end = min(pos + CHUNK, size) - 1
            f.seek(pos)
            data = f.read(end - pos + 1)
            try:
                r = requests.put(url, data=data, timeout=900,
                                 headers={"Content-Length": str(len(data)), "Content-Range": f"bytes {pos}-{end}/{size}"})
                code = r.status_code
            except requests.RequestException as e:
                print("chunk interrupted:", e, flush=True)
                code = None
            if code in (200, 201):
                print(f"upload complete, {size / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)
                return
            if code not in (308, None):
                raise RuntimeError(f"chunk returned {code}: {r.text[:300]}")
            for attempt in range(10):
                try:
                    got = _received(url, size)
                    break
                except (requests.RequestException, RuntimeError) as e:
                    print("status query failed:", e, flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError("upload status unknown")
            if got is None:
                print(f"upload complete, {size / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)
                return
            pos = got
            print(f"  {pos / 1e6:.1f} / {size / 1e6:.1f} MB, {time.time() - t0:.0f}s", flush=True)


def submit(path: str, message: str, competition: str) -> str:
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        req = ApiStartSubmissionUploadRequest()
        req.competition_name = competition
        req.file_name = os.path.basename(path)
        req.content_length = os.path.getsize(path)
        req.last_modified_epoch_seconds = int(os.path.getmtime(path))
        resp = kaggle.competitions.competition_api_client.start_submission_upload(req)
    # kept so that a failed create_submission can be retried without uploading again
    with open(path + ".upload.json", "w") as f:
        json.dump(dict(token=resp.token, url=resp.create_url, time=time.time()), f)
    upload(path, resp.create_url)
    return create(resp.token, message, competition)


def create(token: str, message: str, competition: str) -> str:
    api = KaggleApi()
    api.authenticate()
    for attempt in range(6):
        try:
            with api.build_kaggle_client() as kaggle:
                req = ApiCreateSubmissionRequest()
                req.competition_name = competition
                req.blob_file_tokens = token
                req.submission_description = message
                out = kaggle.competitions.competition_api_client.create_submission(req)
            return str(out.message)
        except Exception as e:  # a dropped connection here must not lose the uploaded file
            print("create_submission failed:", e, flush=True)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("create_submission failed 6 times; token saved next to the file")
