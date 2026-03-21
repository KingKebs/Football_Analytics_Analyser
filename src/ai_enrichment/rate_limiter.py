"""
rate_limiter.py
---------------
Token-bucket rate limiter + exponential backoff for all external API calls.

Rate limits we must respect:
  football-data.org free tier  : 10 req/min  (enforced hard)
  Claude API (Anthropic)       : ~50 req/min default (soft throttle at 40)
  News/injury scrapers         : 1 req/2s    (polite crawl)

Strategy for the enrichment pipeline:
  - ONE batch call to football-data.org fetches ALL fixtures for a date in
    a SINGLE request. Never loop per-fixture. This uses ~1-2 requests total.
  - LLM injury enrichment batches ALL fixtures for a matchday into a SINGLE
    Claude API call (structured JSON output). One call, not N calls.
  - A local file cache (data/cache/) prevents re-fetching on re-runs same day.
  - All calls go through RateLimiter.wait() before executing.
"""

import time
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token-bucket rate limiter. Thread-safe.

    Usage:
        limiter = RateLimiter(calls_per_minute=10)
        limiter.wait()          # blocks until a token is available
        response = requests.get(url)
    """

    def __init__(self, calls_per_minute: int, name: str = "api"):
        self.name = name
        self.calls_per_minute = calls_per_minute
        self._interval = 60.0 / calls_per_minute   # seconds between tokens
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0

    def wait(self) -> None:
        """Block until it is safe to make the next API call."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            if elapsed < self._interval:
                sleep_for = self._interval - elapsed
                logger.debug(
                    "[%s] Rate limit: sleeping %.2fs before next call",
                    self.name, sleep_for
                )
                time.sleep(sleep_for)
            self._last_call_time = time.monotonic()


# -- Singleton limiters used by all enrichment modules -------------------------

# football-data.org: 10 req/min free tier. We target 8 to leave headroom.
FOOTBALL_DATA_LIMITER = RateLimiter(calls_per_minute=8, name="football-data.org")

# Anthropic Claude API: safe throttle at 40 req/min to stay well under limits.
# In practice the enrichment pipeline makes 1-2 Claude calls per workflow run,
# so this is just a safety net.
CLAUDE_LIMITER = RateLimiter(calls_per_minute=40, name="claude-api")

# Generic web scraping (news / injury sites): 1 req/2s = 30 req/min, polite.
WEB_SCRAPE_LIMITER = RateLimiter(calls_per_minute=30, name="web-scrape")


def with_retry(
    fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 5.0,
    limiter: RateLimiter = None,
    **kwargs
):
    """
    Call fn(*args, **kwargs) with exponential backoff on failure.

    Handles:
      - HTTP 429 Too Many Requests  -> doubles delay, retries
      - HTTP 5xx server errors      -> retries with backoff
      - Connection errors           -> retries with backoff

    Args:
        fn:           Callable that makes the API/HTTP call
        max_retries:  Maximum retry attempts (default 3)
        base_delay:   Initial retry delay in seconds (doubles each attempt)
        limiter:      RateLimiter to respect before each attempt

    Returns:
        Whatever fn returns on success.

    Raises:
        Last exception if all retries exhausted.
    """
    import requests

    last_exc = None
    delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            if limiter:
                limiter.wait()

            result = fn(*args, **kwargs)

            # If result is a requests.Response, check status
            if hasattr(result, "status_code"):
                if result.status_code == 429:
                    retry_after = int(result.headers.get("Retry-After", delay))
                    logger.warning(
                        "429 Too Many Requests. Waiting %ds before retry %d/%d",
                        retry_after, attempt + 1, max_retries
                    )
                    time.sleep(retry_after)
                    continue
                if result.status_code >= 500:
                    logger.warning(
                        "HTTP %d server error. Retry %d/%d in %.0fs",
                        result.status_code, attempt + 1, max_retries, delay
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                result.raise_for_status()

            return result

        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            logger.warning(
                "Connection error on attempt %d/%d: %s. Retrying in %.0fs",
                attempt + 1, max_retries, exc, delay
            )
            time.sleep(delay)
            delay *= 2

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                logger.warning(
                    "Error on attempt %d/%d: %s. Retrying in %.0fs",
                    attempt + 1, max_retries, exc, delay
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise

    raise last_exc
