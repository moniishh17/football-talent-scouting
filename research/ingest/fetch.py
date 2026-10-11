"""Resumable, cached access to StatsBomb Open Data with provenance.

Every downloaded file is stored gzip-compressed under data/research/cache/<kind>/ and
recorded in a JSON-lines manifest (url, bytes, sha256, fetched_at_utc). Re-running never
re-downloads a file already present. Raw event JSON is large, so events are *not* kept; the
event pipeline caches only the derived per-player-match aggregates (see events.py) and records
the sha256 of the source event file in the manifest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from research.config import CACHE, OPEN_DATA_BASE, USER_AGENT

_manifest_lock = threading.Lock()
MANIFEST = CACHE / "manifest.jsonl"


def http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except Exception as exc:  # network hiccup: back off and retry
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last}")


def record_manifest(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock, MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def cached_json(kind: str, rel: str, cache_name: str | None = None):
    """Fetch <OPEN_DATA_BASE>/<rel> once; return parsed JSON. kind is a cache subfolder."""
    name = cache_name or rel.replace("/", "_")
    path = CACHE / kind / (name + ".gz")
    if path.exists():
        with gzip.open(path, "rb") as fh:
            return json.loads(fh.read())
    url = f"{OPEN_DATA_BASE}/{rel}"
    raw = http_get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as fh:
        fh.write(raw)
    tmp.replace(path)
    record_manifest({"url": url, "kind": kind, "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return json.loads(raw)
