"""
src/ai_enrichment
-----------------
AI automation integrations for the Football Analytics Analyser.

Branch: feature/ai-enrichment-layer
Mapped to: run_analysis_workflow.py

Modules
-------
rate_limiter         Token-bucket rate limiter + exponential backoff
cache                File-based daily cache (prevents re-hitting APIs on re-runs)
fixtures_fetcher     Integration A -- auto-fetch upcomingMatches.json (1 API call/day)
context_warnings     Integration C -- cup/derby/neutral venue flags in detect_leagues()
explainer            Integration D -- LLM rationale (1 Claude call for all predictions)
value_scanner        Integration E -- value alert scanner (no external calls)
digest               Integration H -- daily HTML/PDF/email digest after Step 6
streamlit_chat_tab   Integration I -- Streamlit chat component (drop-in tab)

Net API calls per workflow run:
  First run of day  : 2  (1x football-data.org + 1x Claude for explainer)
  Re-runs same day  : 0  (both are cache-hit)
  Chat tab          : 1 per user message (interactive, not batch)
"""

from .rate_limiter import RateLimiter, FOOTBALL_DATA_LIMITER, CLAUDE_LIMITER
from .cache import cache_get, cache_set, cache_clear
from .fixtures_fetcher import pre_run_fetch
from .context_warnings import analyse_context_warnings, print_context_warnings
from .explainer import post_process_explanations
from .value_scanner import post_scan_value_bets
from .digest import generate_digest

__all__ = [
    "RateLimiter",
    "FOOTBALL_DATA_LIMITER",
    "CLAUDE_LIMITER",
    "cache_get",
    "cache_set",
    "cache_clear",
    "pre_run_fetch",
    "analyse_context_warnings",
    "print_context_warnings",
    "post_process_explanations",
    "post_scan_value_bets",
    "generate_digest",
]
