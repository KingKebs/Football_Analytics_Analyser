"""
explainer.py  (Integration D)
------------------------------
Appends an 'ai_rationale' field to every prediction in the consolidated
full-league output JSON. Uses a SINGLE Claude API call for the entire
matchday -- not one call per fixture.

RATE LIMIT STRATEGY
-------------------
Anthropic API: ~50 req/min default. We use 1 call per workflow run.

The key design decision is batching: all predictions for the day are
serialised into one prompt, and Claude returns a JSON array of rationales
in one response. This means:

  10 fixtures on a matchday -> 1 API call  (not 10)
  50 fixtures on a matchday -> 1 API call  (not 50)

This is both cheaper and eliminates any risk of hitting Anthropic rate limits.
The only practical limit is the Claude context window (~200k tokens), which
is far larger than any realistic matchday fixture list.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .cache import cache_get, cache_set
from .rate_limiter import CLAUDE_LIMITER

logger = logging.getLogger(__name__)

# Maximum characters of per-match stats sent to Claude
# Keeps prompt lean; full data is in the JSON files for Streamlit
MAX_STATS_CHARS_PER_MATCH = 400


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _build_batch_prompt(predictions: list[dict]) -> str:
    """
    Build a single prompt that asks Claude to explain ALL predictions
    in one structured JSON response.
    """
    # Summarise each match to keep prompt size manageable
    match_summaries = []
    for i, m in enumerate(predictions):
        summary = {
            "idx": i,
            "home": m.get("home_team") or m.get("homeTeam") or m.get("home", "?"),
            "away": m.get("away_team") or m.get("awayTeam") or m.get("away", "?"),
            "league": m.get("league") or m.get("competition", ""),
            "prediction": m.get("prediction") or m.get("suggested_bet", ""),
            "probability": m.get("probability") or m.get("confidence", 0),
            "edge": m.get("edge", 0),
            "kelly": m.get("kelly_fraction") or m.get("kelly", 0),
            "xg_home": m.get("xg_home") or m.get("home_xg", ""),
            "xg_away": m.get("xg_away") or m.get("away_xg", ""),
            "home_form": m.get("home_form", ""),
            "away_form": m.get("away_form", ""),
        }
        # Truncate to keep prompt lean
        summary_str = json.dumps(summary)[:MAX_STATS_CHARS_PER_MATCH]
        match_summaries.append(summary_str)

    matches_block = "\n".join(
        f"Match {i}: {s}" for i, s in enumerate(match_summaries)
    )

    prompt = f"""You are a concise football betting analyst.
For each match below, write a 1-2 sentence rationale explaining WHY this prediction has edge.
Be direct and specific. Reference the actual stats (xG, form, Kelly) if meaningful.
Do not hedge or add disclaimers.

Return ONLY a JSON array with one entry per match, in this exact format:
[
  {{"idx": 0, "rationale": "explanation here"}},
  {{"idx": 1, "rationale": "explanation here"}},
  ...
]

Matches:
{matches_block}
"""
    return prompt


def _parse_batch_response(response_text: str, count: int) -> dict[int, str]:
    """
    Parse Claude's JSON array response into a dict of idx -> rationale.
    Handles markdown code fences if Claude wraps the JSON.
    """
    # Strip ```json ... ``` fences if present
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response_text).strip()

    try:
        items = json.loads(cleaned)
        return {item["idx"]: item["rationale"] for item in items if "idx" in item}
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse Claude batch response: %s", exc)
        logger.debug("Raw response: %s", response_text[:500])
        # Return empty rationales rather than crashing the workflow
        return {}


def post_process_explanations(
    consolidated_path: str,
    date_str: str,
    dry_run: bool = False,
) -> int:
    """
    Main entry point for Integration D.

    Reads the consolidated full-league JSON, appends 'ai_rationale' to
    each prediction using a single Claude API call, and writes it back.

    Call from run_analysis_workflow.py after run_full_league_analysis():

        from src.ai_enrichment.explainer import post_process_explanations
        post_process_explanations(consolidated_path, self.date)

    Args:
        consolidated_path: Path to consolidated_full_league_<DATE>.json
        date_str:          Date in YYYYMMDD format (for cache key)
        dry_run:           If True, skip the API call (for testing)

    Returns:
        Number of predictions enriched.
    """
    logger.info("=== Integration D: post_process_explanations() ===")

    if not Path(consolidated_path).exists():
        logger.error("Consolidated file not found: %s", consolidated_path)
        return 0

    data = _load_json(consolidated_path)
    predictions = data.get("predictions", [])

    if not predictions:
        logger.warning("No predictions found in %s", consolidated_path)
        return 0

    logger.info("Enriching %d predictions with AI rationales", len(predictions))

    # -- Check cache ---------------------------------------------------------
    cache_key = f"explanations_{Path(consolidated_path).stem}"
    cached_rationales = cache_get(cache_key, date_str)
    if cached_rationales is not None:
        logger.info("Using cached rationales for %s", date_str)
        rationale_map = cached_rationales
    elif dry_run:
        logger.info("Dry run -- skipping Claude API call")
        rationale_map = {i: "[dry run -- no API call]" for i in range(len(predictions))}
    else:
        # -- Single batched Claude API call -----------------------------------
        import anthropic
        client = anthropic.Anthropic()
        prompt = _build_batch_prompt(predictions)

        logger.info(
            "Calling Claude API (1 batched call for %d predictions)",
            len(predictions)
        )
        CLAUDE_LIMITER.wait()

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.content[0].text
        rationale_map = _parse_batch_response(response_text, len(predictions))

        # Cache the rationale map
        cache_set(cache_key, date_str, rationale_map)

    # -- Apply rationales to predictions -------------------------------------
    enriched = 0
    for i, prediction in enumerate(predictions):
        rationale = rationale_map.get(i) or rationale_map.get(str(i))
        if rationale:
            prediction["ai_rationale"] = rationale
            enriched += 1
        else:
            prediction["ai_rationale"] = ""

    data["_ai_enrichment"] = {
        "enriched_at": __import__("datetime").datetime.utcnow().isoformat(),
        "model": "claude-sonnet-4-6",
        "predictions_enriched": enriched,
        "api_calls_made": 0 if (cached_rationales or dry_run) else 1,
    }

    _write_json(consolidated_path, data)
    logger.info(
        "Enriched %d/%d predictions. 1 API call used. Wrote back to %s",
        enriched, len(predictions), consolidated_path
    )
    return enriched
