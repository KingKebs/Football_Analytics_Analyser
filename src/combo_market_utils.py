"""
Combo Market Utilities

Companion utilities for combo market integration:
- Value detection in combo markets
- Comparison of combo vs independent leg pricing
- EV analysis for combo bets
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List

from src.algorithms import extract_combo_markets, prob_to_decimal_odds, kelly_fraction

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')


def detect_combo_value(combo_probs: Dict[str, Dict[str, float]],
                       standard_probs: Dict[str, Dict[str, float]],
                       min_ev_threshold: float = 0.05) -> List[Dict]:
    """
    Detect underpriced combo markets by comparing to independent leg pricing.

    Key insight: Bookies often price combos assuming independence.
    If markets are correlated, combos are mispriced.

    Example:
    - Combo odds for "Home Win & Over 2.5" = 2.50
    - Your calc: P(Home) * P(Over2.5) = 0.55 * 0.52 = 0.286 (odds 3.50)
    - If correlation exists: actual prob might be 0.31 (odds 3.23)
    - Bookie gives 2.50 → strong value! EV = (0.31 × 2.50) - 1 = -0.225... wait this is negative

    Actually: If your true prob is 0.31 but bookie gives 2.50 odds:
    - Your implied prob from odds: 1/2.50 = 0.40
    - Your true prob: 0.31
    - Bookie OVERPRICES combo → skip (we want underpriced)

    Better example:
    - Combo odds: 3.80 (implies 26.3%)
    - Your true prob from correlation: 31%
    - EV = (0.31 × 3.80) - 1 = +0.178 → VALUE! ✓

    Args:
        combo_probs: Dictionary of combo market probabilities
        standard_probs: Dictionary of standard market probabilities
        min_ev_threshold: Minimum EV to flag as value opportunity (default 5%)

    Returns:
        List of value opportunities with details
    """
    value_opportunities = []

    # Flatten combo probs for easier processing
    flat_combos = {}
    for combo_type, markets in combo_probs.items():
        if isinstance(markets, dict):
            for threshold_key, selections in markets.items():
                if isinstance(selections, dict):
                    # Nested (e.g., 1X2_OU with multiple thresholds)
                    for sel, prob in selections.items():
                        flat_combos[f"{combo_type}_{threshold_key}_{sel}"] = prob
                else:
                    # Flat (e.g., 1X2_BTTS)
                    for sel, prob in markets.items():
                        flat_combos[f"{combo_type}_{sel}"] = prob

    # Get standard market probs for comparison
    standard_1x2 = standard_probs.get('1X2', {})
    standard_ou = standard_probs.get('OU', {})
    standard_btts = standard_probs.get('BTTS', {})

    # Analyze each combo for value
    for combo_key, combo_prob in flat_combos.items():
        if combo_prob <= 0 or combo_prob >= 1.0:
            continue

        # Parse combo key to identify components
        # Examples: 1X2_OU2.5_Home_Under, 1X2_BTTS_Home_BTTS_Yes, OU_BTTS2.5_Over_BTTS_Yes

        try:
            parts = combo_key.split('_')
            combo_type = parts[0]

            # Skip analysis if we can't parse
            if len(parts) < 3:
                continue

            # Try to reconstruct expected prob from independent markets
            expected_prob = None

            # 1X2 & Over/Under analysis
            if '1X2_OU' in combo_key and len(parts) >= 4:
                # Format: 1X2_OU[threshold]_[outcome]_[direction]
                outcome = parts[3]  # Home, Draw, Away
                direction = parts[4] if len(parts) > 4 else None  # Under, Over

                p_1x2 = standard_1x2.get(outcome, 0)

                # Map direction to OU market
                ou_key = None
                if direction == 'Under' and '1.5' in combo_key:
                    ou_key = 'Under1.5'
                elif direction == 'Over' and '1.5' in combo_key:
                    ou_key = 'Over1.5'
                elif direction == 'Under' and '2.5' in combo_key:
                    ou_key = 'Under2.5'
                elif direction == 'Over' and '2.5' in combo_key:
                    ou_key = 'Over2.5'

                if ou_key:
                    p_ou = standard_ou.get(ou_key, 0)
                    # Assume independent (conservative estimate)
                    expected_prob = p_1x2 * p_ou

            # 1X2 & BTTS analysis
            elif '1X2_BTTS' in combo_key:
                parts_clean = combo_key.replace('1X2_BTTS_', '').split('_')
                outcome = parts_clean[0]  # Home, Draw, Away
                btts = parts_clean[1] if len(parts_clean) > 1 else None  # Yes, No

                p_1x2 = standard_1x2.get(outcome, 0)
                p_btts = standard_btts.get(btts, 0) if btts else 0
                expected_prob = p_1x2 * p_btts

            # Over/Under & BTTS analysis
            elif 'OU_BTTS' in combo_key:
                parts_clean = combo_key.replace('OU_BTTS', '').split('_')
                direction = parts_clean[1] if len(parts_clean) > 1 else None  # Under, Over
                btts = parts_clean[2] if len(parts_clean) > 2 else None  # Yes, No

                ou_key = None
                if direction == 'Under' and '2.5' in combo_key:
                    ou_key = 'Under2.5'
                elif direction == 'Over' and '2.5' in combo_key:
                    ou_key = 'Over2.5'

                if ou_key:
                    p_ou = standard_ou.get(ou_key, 0)
                    p_btts = standard_btts.get(btts, 0) if btts else 0
                    expected_prob = p_ou * p_btts

            # Calculate EV
            if expected_prob and expected_prob > 0:
                combo_odds = prob_to_decimal_odds(combo_prob)
                bookie_implied_prob = 1.0 / combo_odds if combo_odds > 0 else 0

                # EV = (true_prob × odds) - 1
                ev = (combo_prob * combo_odds) - 1.0
                ev_pct = ev * 100

                # Also check for arbitrage (when true prob > implied prob)
                arbitrage_edge = combo_prob - bookie_implied_prob

                # Lower threshold: include if positive EV (min_ev_threshold is very low like 0.005) OR has arbitrage
                if ev_pct >= min_ev_threshold * 100 or arbitrage_edge > 0.01:
                    value_opportunities.append({
                        'combo': combo_key,
                        'combo_probability': combo_prob,
                        'expected_probability': expected_prob,
                        'bookie_implied_prob': bookie_implied_prob,
                        'combo_odds': combo_odds,
                        'ev_percentage': ev_pct,
                        'arbitrage_edge': arbitrage_edge,
                        'kelly_stake': kelly_fraction(combo_prob, combo_odds),
                        'value_type': 'positive_ev' if ev_pct > 0 else 'arbitrage',
                        'recommendation': 'BET' if ev_pct >= min_ev_threshold * 100 else ('MONITOR' if arbitrage_edge > 0.01 else 'PASS')
                    })

        except Exception as e:
            logging.debug(f"Could not analyze combo {combo_key}: {e}")
            continue

    return sorted(value_opportunities, key=lambda x: x['ev_percentage'], reverse=True)


def compare_combo_vs_parlay(combo_odds: float, combo_prob: float,
                            leg1_odds: float, leg1_prob: float,
                            leg2_odds: float, leg2_prob: float) -> Dict:
    """
    Compare combo bet vs parlay of same two legs.

    Combos often have better odds than betting two legs as a parlay.

    Args:
        combo_odds: Decimal odds for combo bet
        combo_prob: Probability of combo outcome
        leg1_odds, leg1_prob: Odds and prob for leg 1
        leg2_odds, leg2_prob: Odds and prob for leg 2

    Returns:
        Comparison with recommendation
    """
    # Parlay calculation
    parlay_prob = leg1_prob * leg2_prob
    parlay_odds = leg1_odds * leg2_odds

    # EV comparison
    combo_ev = (combo_prob * combo_odds) - 1
    parlay_ev = (parlay_prob * parlay_odds) - 1

    # Stake comparison (Kelly)
    combo_stake = kelly_fraction(combo_prob, combo_odds)
    parlay_stake = kelly_fraction(parlay_prob, parlay_odds)

    return {
        'combo': {
            'odds': combo_odds,
            'prob': combo_prob,
            'ev': combo_ev,
            'kelly_stake': combo_stake
        },
        'parlay': {
            'odds': parlay_odds,
            'prob': parlay_prob,
            'ev': parlay_ev,
            'kelly_stake': parlay_stake
        },
        'combo_advantage': {
            'odds_better_by': combo_odds - parlay_odds,
            'ev_better_by': combo_ev - parlay_ev,
            'stake_reduced_by': parlay_stake - combo_stake,
            'recommendation': 'USE_COMBO' if combo_ev > parlay_ev else 'USE_PARLAY'
        }
    }


def print_value_report(value_opps: List[Dict]):
    """Print formatted report of value opportunities in combos."""
    if not value_opps:
        print("❌ No value opportunities detected in combo markets")
        return

    print(f"\n{'='*100}")
    print(f"🎯 COMBO MARKET VALUE OPPORTUNITIES ({len(value_opps)} found)")
    print(f"{'='*100}\n")

    for i, opp in enumerate(value_opps[:10], 1):  # Top 10
        print(f"{i}. {opp['combo']}")
        print(f"   Your Prob: {opp['combo_probability']*100:.1f}% | Bookie Odds: {opp['combo_odds']:.2f} (implies {opp['bookie_implied_prob']*100:.1f}%)")
        print(f"   EV: {opp['ev_percentage']:+.2f}% | Kelly Stake: {opp['kelly_stake']:.4f} (1.5% max)")
        print(f"   ⭐ {opp['recommendation']}")
        print()


if __name__ == "__main__":
    # Example usage
    print("Combo Market Utilities loaded")
    print("Use detect_combo_value() to find underpriced combos")
    print("Use compare_combo_vs_parlay() to compare betting strategies")

