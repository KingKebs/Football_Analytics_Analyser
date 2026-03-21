"""
tests/test_ai_enrichment.py
----------------------------
Unit tests for all Integration A / D / E modules.
Uses mocking -- no real API calls made during tests.

Run:
    python -m pytest tests/test_ai_enrichment.py -v
"""

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ai_enrichment.rate_limiter import RateLimiter
from src.ai_enrichment.cache import cache_get, cache_set, cache_clear
from src.ai_enrichment.value_scanner import post_scan_value_bets, _score_alert


# -----------------------------------------------------------------------------
# RateLimiter tests
# -----------------------------------------------------------------------------

class TestRateLimiter:
    def test_throttles_calls(self):
        """Two rapid calls to a 60/min limiter should space ~1s apart."""
        limiter = RateLimiter(calls_per_minute=60, name="test")
        t0 = time.monotonic()
        limiter.wait()
        limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.9, f"Expected >=0.9s between calls, got {elapsed:.2f}s"

    def test_no_sleep_if_interval_passed(self):
        """If enough time has passed, wait() should not block."""
        limiter = RateLimiter(calls_per_minute=600, name="test-fast")  # 0.1s interval
        limiter.wait()
        time.sleep(0.15)
        t0 = time.monotonic()
        limiter.wait()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, f"Should not have slept, but elapsed {elapsed:.2f}s"


# -----------------------------------------------------------------------------
# Cache tests
# -----------------------------------------------------------------------------

class TestCache:
    def test_roundtrip(self, tmp_path, monkeypatch):
        """set -> get should return original data."""
        monkeypatch.chdir(tmp_path)
        cache_set("test_key", "20260315", {"predictions": [1, 2, 3]})
        result = cache_get("test_key", "20260315")
        assert result == {"predictions": [1, 2, 3]}

    def test_miss_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert cache_get("nonexistent", "20260315") is None

    def test_expired_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cache_set("expiry_test", "20260315", {"x": 1})

        # Manually backdating the stored_at time
        cache_path = tmp_path / "data" / "cache" / "enrichment_expiry_test_20260315.json"
        with cache_path.open() as f:
            envelope = json.load(f)
        envelope["stored_at"] = time.time() - (24 * 3600)  # 24h ago
        with cache_path.open("w") as f:
            json.dump(envelope, f)

        assert cache_get("expiry_test", "20260315") is None

    def test_clear(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cache_set("k1", "20260315", {"a": 1})
        cache_set("k2", "20260315", {"b": 2})
        deleted = cache_clear("20260315")
        assert deleted == 2
        assert cache_get("k1", "20260315") is None


# -----------------------------------------------------------------------------
# Value scanner tests
# -----------------------------------------------------------------------------

class TestValueScanner:
    def _make_consolidated(self, predictions: list) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump({"predictions": predictions}, tmp)
        tmp.close()
        return tmp.name

    def test_filters_below_thresholds(self, tmp_path):
        predictions = [
            {"home": "A", "away": "B", "edge": 0.02, "kelly_fraction": 0.05, "probability": 0.6},  # edge too low
            {"home": "C", "away": "D", "edge": 0.08, "kelly_fraction": 0.01, "probability": 0.6},  # kelly too low
            {"home": "E", "away": "F", "edge": 0.08, "kelly_fraction": 0.05, "probability": 0.6},  # PASSES
        ]
        path = self._make_consolidated(predictions)
        out = post_scan_value_bets(path, "20260315", output_dir=str(tmp_path))
        result = json.loads(Path(out).read_text())
        assert result["_meta"]["alerts_found"] == 1
        assert result["alerts"][0]["home"] == "E"

    def test_sorted_by_score(self, tmp_path):
        predictions = [
            {"home": "A", "away": "B", "edge": 0.06, "kelly_fraction": 0.04, "probability": 0.55},
            {"home": "C", "away": "D", "edge": 0.12, "kelly_fraction": 0.08, "probability": 0.65},
            {"home": "E", "away": "F", "edge": 0.09, "kelly_fraction": 0.06, "probability": 0.60},
        ]
        path = self._make_consolidated(predictions)
        out = post_scan_value_bets(path, "20260315", output_dir=str(tmp_path))
        result = json.loads(Path(out).read_text())
        # Should be sorted highest score first
        edges = [a["edge"] for a in result["alerts"]]
        assert edges == sorted(edges, reverse=True)

    def test_caps_kelly(self, tmp_path):
        predictions = [
            {"home": "A", "away": "B", "edge": 0.15, "kelly_fraction": 0.30, "probability": 0.70},  # kelly > 0.25 cap
        ]
        path = self._make_consolidated(predictions)
        out = post_scan_value_bets(path, "20260315", output_dir=str(tmp_path))
        result = json.loads(Path(out).read_text())
        assert result["_meta"]["alerts_found"] == 0

    def test_empty_predictions(self, tmp_path):
        path = self._make_consolidated([])
        out = post_scan_value_bets(path, "20260315", output_dir=str(tmp_path))
        result = json.loads(Path(out).read_text())
        assert result["_meta"]["alerts_found"] == 0


# -----------------------------------------------------------------------------
# Fixtures fetcher tests (mocked)
# -----------------------------------------------------------------------------

class TestFixturesFetcher:
    MOCK_API_RESPONSE = {
        "matches": [
            {
                "id": 123,
                "competition": {"code": "PL", "name": "Premier League"},
                "utcDate": "2026-03-15T15:00:00Z",
                "homeTeam": {"name": "Arsenal FC"},
                "awayTeam": {"name": "Chelsea FC"},
                "status": "SCHEDULED",
            },
            {
                "id": 124,
                "competition": {"code": "UNKNOWN", "name": "Unknown League"},
                "utcDate": "2026-03-15T17:00:00Z",
                "homeTeam": {"name": "Team A"},
                "awayTeam": {"name": "Team B"},
                "status": "SCHEDULED",
            },
        ]
    }

    def test_transforms_and_filters(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.MOCK_API_RESPONSE

        with patch("src.ai_enrichment.fixtures_fetcher.requests.get", return_value=mock_response):
            from src.ai_enrichment.fixtures_fetcher import fetch_fixtures_for_date
            fixtures = fetch_fixtures_for_date("2026-03-15")

        # UNKNOWN competition should be filtered out
        assert len(fixtures) == 1
        assert fixtures[0]["leagueCode"] == "E0"
        assert fixtures[0]["homeTeam"] == "Arsenal FC"
        assert fixtures[0]["time"] == "15:00"

    def test_cache_prevents_second_call(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.MOCK_API_RESPONSE

        with patch("src.ai_enrichment.fixtures_fetcher.requests.get", return_value=mock_response) as mock_get:
            from src.ai_enrichment.fixtures_fetcher import fetch_fixtures_for_date
            fetch_fixtures_for_date("2026-03-15")
            fetch_fixtures_for_date("2026-03-15")   # second call -- should use cache
            assert mock_get.call_count == 1, "Second call should hit cache, not API"


# -----------------------------------------------------------------------------
# Explainer tests (mocked)
# -----------------------------------------------------------------------------

class TestExplainer:
    def test_dry_run_appends_fields(self, tmp_path):
        consolidated = tmp_path / "consolidated_test.json"
        data = {
            "predictions": [
                {"home": "Arsenal", "away": "Chelsea", "prediction": "Home Win",
                 "probability": 0.62, "edge": 0.08, "kelly_fraction": 0.05},
                {"home": "Liverpool", "away": "Man City", "prediction": "Draw",
                 "probability": 0.38, "edge": 0.06, "kelly_fraction": 0.03},
            ]
        }
        consolidated.write_text(json.dumps(data))

        from src.ai_enrichment.explainer import post_process_explanations
        enriched = post_process_explanations(
            str(consolidated), "20260315", dry_run=True
        )

        assert enriched == 2
        result = json.loads(consolidated.read_text())
        for pred in result["predictions"]:
            assert "ai_rationale" in pred
        assert "_ai_enrichment" in result

    def test_missing_file_returns_zero(self):
        from src.ai_enrichment.explainer import post_process_explanations
        count = post_process_explanations("/nonexistent/path.json", "20260315")
        assert count == 0
