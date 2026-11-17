"""HTTP access with a small on-disk cache and NVD rate limiting.

Vulnerability data changes daily, so cached responses expire after ``ttl`` seconds (one day by
default). The cache keeps repeated questions and evaluation runs fast and within the NVD limits
(5 requests per 30 s without an API key, 50 with one; set ``NVD_API_KEY`` to use a key).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import requests

CACHE_DIR = Path(os.environ.get("VULNSCOUT_CACHE", Path(__file__).resolve().parent.parent / "data" / "cache"))
DAY = 24 * 3600
_session = requests.Session()
_session.headers["User-Agent"] = "VulnScout/0.1 (INFO 7375 course project)"


class _RateLimiter:
    def __init__(self, calls: int, period: float):
        self.calls, self.period = calls, period
        self.stamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self.stamps and now - self.stamps[0] > self.period:
            self.stamps.popleft()
        if len(self.stamps) >= self.calls:
            time.sleep(self.period - (now - self.stamps[0]) + 0.1)
        self.stamps.append(time.monotonic())


_nvd_limit = _RateLimiter(45 if os.environ.get("NVD_API_KEY") else 5, 30.0)


def _cache_path(url: str, params: dict | None, body: dict | None) -> Path:
    key = json.dumps([url, params, body], sort_keys=True)
    return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def fetch_json(url: str, params: dict | None = None, body: dict | None = None,
               ttl: float = DAY, timeout: float = 30.0) -> dict:
    """GETs (or POSTs ``body``) and returns parsed JSON, using the cache when it is fresh."""
    path = _cache_path(url, params, body)
    if ttl > 0 and path.exists() and time.time() - path.stat().st_mtime < ttl:
        return json.loads(path.read_text(encoding="utf-8"))

    headers = {}
    is_nvd = "services.nvd.nist.gov" in url
    if is_nvd and os.environ.get("NVD_API_KEY"):
        headers["apiKey"] = os.environ["NVD_API_KEY"]
    for attempt in range(4):
        if is_nvd:
            _nvd_limit.wait()
        if body is None:
            resp = _session.get(url, params=params, headers=headers, timeout=timeout)
        else:
            resp = _session.post(url, json=body, headers=headers, timeout=timeout)
        if resp.status_code in (403, 429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(6 * (attempt + 1))  # NVD answers 403 when the rate limit is exceeded
            continue
        resp.raise_for_status()
        break
    data = resp.json()
    if ttl > 0:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return data
