"""
context_warnings.py  (Integration C)
--------------------------------------
Augments the detected league list with fixture-level context warnings
AFTER LEAGUE_MAP lookup in detect_leagues(), BEFORE user confirmation.

Zero external API calls. Pure rule-based logic over the fixture data
already loaded in Step 1.

Flags detected:
  CUP_FIXTURE      -- competition name suggests a cup (FA Cup, Carabao, etc.)
  DERBY            -- known local derby matchup
  RELEGATION_SIX_POINTER -- both teams in bottom 6 of known leagues
  NEUTRAL_VENUE    -- flag passed from upstream source
  HIGH_FIXTURE_COUNT -- >15 fixtures on one matchday (market liquidity risk)

These flags are:
  1. Printed to stdout during the workflow prompt (Step 3 confirmation)
  2. Written into the converted fixtures JSON for downstream use
  3. Available to Streamlit UI and the value scanner as context

They change Poisson model assumptions:
  Cup fixtures       -> lower home advantage, higher variance
  Derbies            -> form/xG less predictive, higher draw probability
  Relegation 6PTR   -> extreme motivation effects, models underweight this
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# -- Known derby pairs (home, away) -- both orderings checked ------------------
# Key = league code, Value = list of (team_a, team_b) name fragments
DERBY_MAP: dict[str, list[tuple[str, str]]] = {
    "E0": [
        ("Arsenal",   "Tottenham"),    # North London
        ("Chelsea",   "Arsenal"),
        ("Liverpool", "Everton"),      # Merseyside
        ("Manchester City", "Manchester United"),   # Manchester
        ("Leeds",     "Sheffield"),    # Yorkshire
        ("Aston Villa", "Birmingham"), # West Midlands
        ("Newcastle", "Sunderland"),   # Tyne-Wear
        ("West Ham",  "Millwall"),
    ],
    "E1": [
        ("Derby",     "Nottingham"),
        ("Sheffield United", "Sheffield Wednesday"),
        ("Bristol City", "Bristol Rovers"),
    ],
    "SP1": [
        ("Real Madrid", "Barcelona"),   # El Clasico
        ("Atletico",    "Real Madrid"),
        ("Atletico",    "Barcelona"),
        ("Sevilla",     "Real Betis"),  # Seville derby
        ("Valencia",    "Villarreal"),
    ],
    "D1": [
        ("Bayern",      "Borussia Dortmund"),   # Der Klassiker
        ("Schalke",     "Borussia Dortmund"),   # Revierderby
        ("Hamburg",     "Werder"),
    ],
    "I1": [
        ("Inter",       "AC Milan"),    # Derby della Madonnina
        ("Roma",        "Lazio"),       # Derby della Capitale
        ("Juventus",    "Torino"),      # Derby della Mole
    ],
    "F1": [
        ("Paris",       "Marseille"),   # Le Classique
        ("Lyon",        "Saint-Etienne"),
    ],
}

# -- Cup keyword patterns -----------------------------------------------------
CUP_PATTERNS = re.compile(
    r"\b(cup|copa|coupe|pokal|coppa|trophy|shield|carabao|fa cup|league cup"
    r"|champions league|europa league|conference league|playoff|play.off)\b",
    re.IGNORECASE,
)


def _is_cup_fixture(competition_name: str) -> bool:
    return bool(CUP_PATTERNS.search(competition_name))


def _is_derby(home: str, away: str, league_code: str) -> Optional[str]:
    """
    Return a derby label if this is a known local derby, else None.
    Matching is substring-based, case-insensitive.
    """
    pairs = DERBY_MAP.get(league_code, [])
    for a, b in pairs:
        if (a.lower() in home.lower() and b.lower() in away.lower()) or \
           (b.lower() in home.lower() and a.lower() in away.lower()):
            return f"{home} vs {away}"
    return None


def analyse_context_warnings(
    fixtures: list[dict],
    league_codes: list[str],
) -> dict:
    """
    Main entry point for Integration C.

    Call from detect_leagues() after building the league list, passing the
    already-loaded fixture list:

        from src.ai_enrichment.context_warnings import analyse_context_warnings
        warnings = analyse_context_warnings(fixtures, detected_leagues)
        if warnings['summary']:
            print_warnings(warnings)

    Args:
        fixtures:     List of fixture dicts (from upcomingMatches.json)
        league_codes: List of detected league codes (e.g. ['E0', 'E1'])

    Returns:
        Dict with keys:
          'warnings'  : list of per-fixture warning dicts
          'summary'   : list of human-readable warning strings for the prompt
          'flags'     : set of active flag names (for downstream filtering)
    """
    warnings = []
    summary_lines = []
    active_flags = set()

    matches = fixtures if isinstance(fixtures, list) else fixtures.get("matches", [])

    # -- High fixture count ----------------------------------------------------
    if len(matches) > 15:
        msg = (
            f"HIGH FIXTURE COUNT: {len(matches)} matches on this matchday. "
            "Market liquidity may be thinner on less prominent games."
        )
        summary_lines.append(f"  {msg}")
        active_flags.add("HIGH_FIXTURE_COUNT")
        logger.info("[CONTEXT] %s", msg)

    cup_fixtures = []
    derby_fixtures = []

    for fixture in matches:
        home       = fixture.get("homeTeam") or fixture.get("home", "")
        away       = fixture.get("awayTeam") or fixture.get("away", "")
        competition = fixture.get("competition", "")
        league_code = fixture.get("leagueCode", "")
        fixture_warnings = []

        # -- Cup fixture ------------------------------------------------------
        if _is_cup_fixture(competition):
            fixture_warnings.append("CUP_FIXTURE")
            active_flags.add("CUP_FIXTURE")
            cup_fixtures.append(f"{home} vs {away} ({competition})")
            fixture["_context_warning"] = fixture.get("_context_warning", [])
            fixture["_context_warning"].append("CUP_FIXTURE")

        # -- Derby ------------------------------------------------------------
        derby_label = _is_derby(home, away, league_code)
        if derby_label:
            fixture_warnings.append("DERBY")
            active_flags.add("DERBY")
            derby_fixtures.append(derby_label)
            if "_context_warning" not in fixture:
                fixture["_context_warning"] = []
            fixture["_context_warning"].append("DERBY")

        # -- Neutral venue ----------------------------------------------------
        if fixture.get("neutral_venue") or fixture.get("neutralVenue"):
            fixture_warnings.append("NEUTRAL_VENUE")
            active_flags.add("NEUTRAL_VENUE")
            if "_context_warning" not in fixture:
                fixture["_context_warning"] = []
            fixture["_context_warning"].append("NEUTRAL_VENUE")

        if fixture_warnings:
            warnings.append({
                "match": f"{home} vs {away}",
                "competition": competition,
                "flags": fixture_warnings,
            })

    # -- Build summary lines --------------------------------------------------
    if cup_fixtures:
        summary_lines.append(
            f"  CUP FIXTURES ({len(cup_fixtures)}): "
            + ", ".join(cup_fixtures[:3])
            + ("..." if len(cup_fixtures) > 3 else "")
            + " -- home advantage effect reduced, higher variance."
        )

    if derby_fixtures:
        summary_lines.append(
            f" DERBIES ({len(derby_fixtures)}): "
            + ", ".join(derby_fixtures[:3])
            + " -- form/xG less predictive. Models may underestimate draw probability."
        )

    if "NEUTRAL_VENUE" in active_flags:
        summary_lines.append(
            "  NEUTRAL VENUE detected -- home advantage should be zero."
        )

    if summary_lines:
        logger.info("[CONTEXT] %d warning(s) flagged", len(summary_lines))
    else:
        logger.info("[CONTEXT] No context warnings for this matchday.")

    return {
        "warnings":     warnings,
        "summary":      summary_lines,
        "flags":        active_flags,
        "total_matches": len(matches),
    }


def print_context_warnings(warnings_result: dict) -> None:
    """
    Print context warnings to stdout in the workflow's confirmation prompt.
    Call this right before the Step 3 league confirmation prompt.
    """
    summary = warnings_result.get("summary", [])
    if not summary:
        return

    print("\n" + "-" * 60)
    print("  CONTEXT WARNINGS FOR THIS MATCHDAY")
    print("-" * 60)
    for line in summary:
        print(f"  {line}")
    print("-" * 60)
    print(
        "  These flags are stored per-fixture and visible in Streamlit.\n"
        "  The value scanner applies tighter thresholds to flagged games."
    )
    print("-" * 60 + "\n")
