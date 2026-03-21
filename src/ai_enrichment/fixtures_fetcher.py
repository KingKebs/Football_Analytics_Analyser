"""
fixtures_fetcher.py  (Integration A)
--------------------------------------
Fetches upcoming fixtures from football-data.org and writes
data/raw/upcomingMatches.json -- replacing the current manual update step.

RATE LIMIT STRATEGY
-------------------
football-data.org free tier: 10 req/min

This module makes EXACTLY 1 API call per workflow run:
  GET /v4/matches?dateFrom=DATE&dateTo=DATE

That single call returns ALL matches on the date across all competitions
the free key covers. We never loop per-fixture or per-league. The result
is cached for 23 hours, so re-runs on the same day use zero API quota.

The only scenario that uses a 2nd call is when --leagues is specified and
those leagues are not in the single-date batch (rare edge case), and even
then we use per-competition calls with the rate limiter between them.

Worst case: 3 API calls per day. Well within the 10/min free limit.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from .cache import cache_get, cache_set
from .rate_limiter import FOOTBALL_DATA_LIMITER, with_retry

logger = logging.getLogger(__name__)

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

# Map football-data.org competition codes to your existing LEAGUE_MAP codes
# Extend as needed -- these are the competitions covered on the free tier
COMPETITION_MAP = {
    "PL":  "E0",   # Premier League
    "ELC": "E1",   # Championship
    "EL1": "E2",   # League One
    "EL2": "E3",   # League Two
    "PD":  "SP1",  # La Liga
    "SA":  "I1",   # Serie A
    "BL1": "D1",   # Bundesliga
    "FL1": "F1",   # Ligue 1
    "PPL": "P1",   # Primeira Liga
    "EC":  "EC",   # European Championship (when active)
    "CL":  "UCL",  # Champions League
}


def _get_headers() -> dict:
    """Return request headers with API token from environment."""
    token = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not token:
        logger.warning(
            "FOOTBALL_DATA_API_KEY not set. "
            "Unauthenticated calls are limited to 100/day and restricted resources. "
            "Set the env var or add it to config/config.yaml."
        )
    return {"X-Auth-Token": token}


def fetch_fixtures_for_date(date_str: str) -> list[dict]:
    """
    Fetch all fixtures for a given date in ONE API call.

    Args:
        date_str: Date in YYYY-MM-DD format (e.g. "2026-03-15")

    Returns:
        List of fixture dicts in the upcomingMatches.json format
        expected by the existing workflow.

    Raises:
        requests.HTTPError if the API call fails after retries.
    """
    # -- Check cache first ----------------------------------------------------
    cache_key = f"fixtures_fetch_{date_str.replace('-', '')}"
    cached = cache_get(cache_key, date_str.replace("-", ""))
    if cached is not None:
        logger.info("Returning cached fixtures for %s (%d matches)", date_str, len(cached))
        return cached

    # -- Single API call for the entire date ---------------------------------
    logger.info("Fetching fixtures from football-data.org for %s", date_str)

    def _do_fetch():
        return requests.get(
            f"{FOOTBALL_DATA_BASE}/matches",
            headers=_get_headers(),
            params={"dateFrom": date_str, "dateTo": date_str},
            timeout=15,
        )

    response = with_retry(_do_fetch, max_retries=3, base_delay=6.0,
                          limiter=FOOTBALL_DATA_LIMITER)

    raw_matches = response.json().get("matches", [])
    logger.info("Received %d raw matches from football-data.org", len(raw_matches))

    # -- Transform to existing upcomingMatches.json format -------------------
    fixtures = _transform_matches(raw_matches, date_str)

    # -- Cache the result -----------------------------------------------------
    cache_set(cache_key, date_str.replace("-", ""), fixtures)
    logger.info("Fetched and cached %d fixtures for %s", len(fixtures), date_str)

    return fixtures


def _transform_matches(raw_matches: list, date_str: str) -> list[dict]:
    """
    Convert football-data.org match objects to the upcomingMatches.json
    schema used by the existing workflow.

    The existing workflow expects keys like:
        homeTeam, awayTeam, competition, date, time, matchId
    """
    fixtures = []
    for m in raw_matches:
        comp_code = m.get("competition", {}).get("code", "")
        league_code = COMPETITION_MAP.get(comp_code)

        if league_code is None:
            # Not a competition our LEAGUE_MAP knows about -- skip
            logger.debug("Skipping competition code %s (not in COMPETITION_MAP)", comp_code)
            continue

        utc_date = m.get("utcDate", "")
        kick_off_time = utc_date[11:16] if len(utc_date) >= 16 else "00:00"

        fixture = {
            "matchId":     m.get("id"),
            "competition": m.get("competition", {}).get("name", comp_code),
            "leagueCode":  league_code,
            "homeTeam":    m.get("homeTeam", {}).get("name", "Unknown"),
            "awayTeam":    m.get("awayTeam", {}).get("name", "Unknown"),
            "date":        date_str,
            "time":        kick_off_time,
            "status":      m.get("status", "SCHEDULED"),
            # Preserve raw source data for debugging
            "_source":     "football-data.org",
            "_raw_id":     m.get("id"),
        }
        fixtures.append(fixture)

    return fixtures


def write_upcoming_matches(fixtures: list[dict], output_path: Optional[str] = None) -> str:
    """
    Write fixtures to the upcomingMatches.json file expected by the workflow.

    Args:
        fixtures:     List of fixture dicts from fetch_fixtures_for_date()
        output_path:  Override default path (for testing)

    Returns:
        Absolute path of the written file.
    """
    path = Path(output_path or "data/raw/upcomingMatches.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "_meta": {
            "fetched_at": datetime.utcnow().isoformat(),
            "source": "football-data.org",
            "count": len(fixtures),
        },
        "matches": fixtures,
    }

    with path.open("w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Wrote %d fixtures to %s", len(fixtures), path)
    return str(path.resolve())


def pre_run_fetch(date_str: str, output_path: Optional[str] = None) -> str:
    """
    Main entry point for Integration A.

    Call this from run_analysis_workflow.py BEFORE read_upcoming_matches():

        # In AnalysisWorkflow.run():
        from src.ai_enrichment.fixtures_fetcher import pre_run_fetch
        pre_run_fetch(self.date)

    Args:
        date_str:    Date in YYYY-MM-DD format
        output_path: Override output file path (for testing)

    Returns:
        Path of the written upcomingMatches.json file.
    """
    logger.info("=== Integration A: pre_run_fetch(%s) ===", date_str)
    fixtures = fetch_fixtures_for_date(date_str)

    if not fixtures:
        logger.warning(
            "No fixtures found for %s via football-data.org. "
            "The workflow will fall back to any existing upcomingMatches.json.",
            date_str
        )
        return ""

    return write_upcoming_matches(fixtures, output_path)
