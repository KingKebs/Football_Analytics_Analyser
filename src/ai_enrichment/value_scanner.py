"""
value_scanner.py  (Integration E)
----------------------------------
Scans the consolidated full-league output for high-edge value bets
and writes a ranked shortlist to data/analysis/value_alerts_<DATE>.json.

No external API calls. Zero rate limit exposure.
Purely post-processes the JSON already produced by the workflow.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# -- Default thresholds (can be overridden via config or call args) ----------
DEFAULT_THRESHOLDS = {
    "min_edge":           0.05,   # 5% minimum edge over implied probability
    "min_kelly":          0.02,   # 2% minimum Kelly fraction
    "max_kelly":          0.25,   # Cap -- anything higher may be model noise
    "min_probability":    0.45,   # Minimum prediction confidence
}


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _score_alert(prediction: dict) -> float:
    """
    Composite score for ranking alerts.
    Weights edge most heavily, then Kelly, then probability.
    """
    edge        = prediction.get("edge", 0) or 0
    kelly       = prediction.get("kelly_fraction") or prediction.get("kelly", 0) or 0
    probability = prediction.get("probability") or prediction.get("confidence", 0) or 0
    return (edge * 0.5) + (kelly * 0.3) + (probability * 0.2)


def post_scan_value_bets(
    consolidated_path: str,
    date_str: str,
    thresholds: Optional[dict] = None,
    output_dir: str = "data/analysis",
) -> str:
    """
    Main entry point for Integration E.

    Scans consolidated_full_league_<DATE>.json, filters by edge/Kelly thresholds,
    ranks by composite score, and writes value_alerts_<DATE>.json.

    Call from run_analysis_workflow.py after post_process_explanations():

        from src.ai_enrichment.value_scanner import post_scan_value_bets
        alerts_path = post_scan_value_bets(consolidated_path, self.date)

    Args:
        consolidated_path: Path to consolidated_full_league_<DATE>.json
        date_str:          Date string in YYYYMMDD format
        thresholds:        Override DEFAULT_THRESHOLDS (partial override is OK)
        output_dir:        Directory for value_alerts JSON output

    Returns:
        Path to the written value_alerts_<DATE>.json file.
    """
    logger.info("=== Integration E: post_scan_value_bets() ===")

    if not Path(consolidated_path).exists():
        logger.error("Consolidated file not found: %s", consolidated_path)
        return ""

    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    data = _load_json(consolidated_path)
    predictions = data.get("predictions", [])

    if not predictions:
        logger.warning("No predictions to scan in %s", consolidated_path)

    # -- Filter ---------------------------------------------------------------
    alerts = []
    rejected_reasons: dict[str, int] = {}

    for pred in predictions:
        edge        = pred.get("edge", 0) or 0
        kelly       = pred.get("kelly_fraction") or pred.get("kelly", 0) or 0
        probability = pred.get("probability") or pred.get("confidence", 0) or 0

        if edge < t["min_edge"]:
            rejected_reasons["edge_below_min"] = rejected_reasons.get("edge_below_min", 0) + 1
            continue
        if kelly < t["min_kelly"]:
            rejected_reasons["kelly_below_min"] = rejected_reasons.get("kelly_below_min", 0) + 1
            continue
        if kelly > t["max_kelly"]:
            rejected_reasons["kelly_above_max"] = rejected_reasons.get("kelly_above_max", 0) + 1
            continue
        if probability < t["min_probability"]:
            rejected_reasons["probability_below_min"] = rejected_reasons.get("probability_below_min", 0) + 1
            continue

        alerts.append(pred)

    # -- Rank -----------------------------------------------------------------
    alerts.sort(key=_score_alert, reverse=True)

    # -- Build output ---------------------------------------------------------
    output = {
        "_meta": {
            "generated_at": datetime.utcnow().isoformat(),
            "date": date_str,
            "source_file": consolidated_path,
            "thresholds_applied": t,
            "total_predictions_scanned": len(predictions),
            "alerts_found": len(alerts),
            "rejection_reasons": rejected_reasons,
        },
        "alerts": alerts,
    }

    out_path = str(Path(output_dir) / f"value_alerts_{date_str}.json")
    _write_json(out_path, output)

    logger.info(
        "[VALUE SCAN] %d/%d predictions passed thresholds -> %s",
        len(alerts), len(predictions), out_path
    )
    if rejected_reasons:
        logger.info("[VALUE SCAN] Rejection breakdown: %s", rejected_reasons)

    return out_path
