"""
cache.py
--------
Simple file-based JSON cache keyed by (cache_key, date).

Why this matters for rate limiting:
  If the workflow is run twice in a day (e.g. debug re-run, cron retry),
  cached responses are returned instantly without touching any external API.
  This is the single most effective measure against accidental rate-limit
  exhaustion.

Cache layout:
  data/cache/
    enrichment_<key>_<YYYYMMDD>.json

TTL: cache entries expire after 23 hours (a full matchday window).
     Stale entries are silently ignored and re-fetched.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_TTL_SECONDS = 23 * 3600   # 23 hours


def _cache_path(key: str, date_str: str) -> Path:
    """Return the cache file path for a given key + date."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"enrichment_{safe_key}_{date_str}.json"


def cache_get(key: str, date_str: str) -> Optional[Any]:
    """
    Retrieve a cached value if it exists and has not expired.

    Returns:
        Parsed JSON value, or None if missing / expired / corrupt.
    """
    path = _cache_path(key, date_str)
    if not path.exists():
        return None

    try:
        with path.open() as f:
            envelope = json.load(f)

        stored_at = envelope.get("stored_at", 0)
        if time.time() - stored_at > CACHE_TTL_SECONDS:
            logger.debug("Cache expired for key=%s date=%s", key, date_str)
            path.unlink(missing_ok=True)
            return None

        logger.info("Cache HIT: key=%s date=%s", key, date_str)
        return envelope["data"]

    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Corrupt cache entry %s: %s", path, exc)
        path.unlink(missing_ok=True)
        return None


def cache_set(key: str, date_str: str, data: Any) -> None:
    """
    Store data in the cache for key + date.

    Args:
        key:      Logical cache key (e.g. "fixtures_fetch", "injury_enrichment")
        date_str: Date string in YYYYMMDD format
        data:     JSON-serialisable value to store
    """
    path = _cache_path(key, date_str)
    envelope = {
        "stored_at": time.time(),
        "stored_human": datetime.utcnow().isoformat(),
        "key": key,
        "date": date_str,
        "data": data,
    }
    with path.open("w") as f:
        json.dump(envelope, f, indent=2)
    logger.info("Cache WRITE: key=%s date=%s -> %s", key, date_str, path)


def cache_clear(date_str: Optional[str] = None) -> int:
    """
    Clear cache entries. If date_str given, clears only that date.

    Returns:
        Number of files deleted.
    """
    if not CACHE_DIR.exists():
        return 0

    pattern = f"enrichment_*_{date_str}.json" if date_str else "enrichment_*.json"
    deleted = 0
    for f in CACHE_DIR.glob(pattern):
        f.unlink()
        deleted += 1

    logger.info("Cache cleared: %d file(s) removed", deleted)
    return deleted
