"""
Niche Markets Module

Specialized prediction algorithms for:
- Odd/Even total goals
- Highest Scoring Half (1st vs 2nd)
- Combined parlay optimization
- Lower-league prioritization and boosting

These markets exploit structural match dynamics rather than pure goal volume.

Key Findings (Empirical Analysis):
- English League One (E2): 54.07% odd rate, 1.260 half ratio - TOP PRIORITY
- Spain La Liga 2 (SP2): 1.368 half ratio (37% more 2nd half goals!)
- Greece Super League (G1): 52.93% odd rate
- Netherlands Eredivisie (N1): 1.332 half ratio

Priority: LOWER DIVISIONS > ELITE LEAGUES
"""

from .odd_even_predictor import OddEvenPredictor, compare_leagues as compare_odd_even_leagues
from .half_comparison_predictor import HalfComparisonPredictor, compare_leagues as compare_half_leagues
from .league_priors import get_league_prior, LEAGUE_PRIORS, calculate_league_priors_from_data, get_adaptive_prior
from .lower_league_analysis import LowerLeagueAnalyzer

__all__ = [
    'OddEvenPredictor',
    'HalfComparisonPredictor',
    'LowerLeagueAnalyzer',
    'get_league_prior',
    'get_adaptive_prior',
    'calculate_league_priors_from_data',
    'LEAGUE_PRIORS',
    'compare_odd_even_leagues',
    'compare_half_leagues',
]

