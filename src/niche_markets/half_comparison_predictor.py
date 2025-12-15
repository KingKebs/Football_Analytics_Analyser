"""
Highest Scoring Half Predictor with Lower-League Prioritization

Predicts whether 1st Half or 2nd Half will have more goals, with:
- Time-segmented Poisson distribution
- League-specific half ratios (2nd/1st)
- Fatigue and tactical adjustment modeling
- LOWER DIVISION BOOSTING (SP2, N1, E2 prioritized)

Empirical findings:
- Spain La Liga 2 (SP2): 1.368 ratio (37% more 2nd half goals!) - HIGHEST
- Netherlands Eredivisie (N1): 1.332 ratio
- English League One (E2): 1.260 ratio
- vs Germany Bundesliga (D1): 1.141 ratio (most evenly distributed)
"""

import math
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HalfComparisonPredictor:
    """Predict highest scoring half with league-aware adjustments"""

    # Empirically derived half ratios (2nd half goals / 1st half goals)
    LEAGUE_HALF_RATIOS = {
        'SP2': 1.368,  # Spain La Liga 2 - HIGHEST
        'N1': 1.332,   # Netherlands Eredivisie
        'E0': 1.295,   # England Premier League
        'I1': 1.291,   # Italy Serie A
        'SC0': 1.285,  # Scotland Premiership
        'T1': 1.283,   # Turkey Super Lig
        'F1': 1.282,   # France Ligue 1
        'E2': 1.260,   # England League One
        'G1': 1.253,   # Greece Super League
        'D2': 1.242,   # Germany 2. Bundesliga
        'P1': 1.233,   # Portugal Primeira Liga
        'F2': 1.233,   # France Ligue 2
        'E3': 1.212,   # England League Two
        'I2': 1.196,   # Italy Serie B
        'E1': 1.182,   # England Championship
        'B1': 1.172,   # Belgium Pro League
        'SC2': 1.172,  # Scotland League One
        'SC1': 1.164,  # Scotland Championship
        'SC3': 1.162,  # Scotland League Two
        'D1': 1.141,   # Germany Bundesliga - LOWEST (most evenly distributed)
        'SP1': 1.242,  # Spain La Liga
    }

    # Priority tiers based on 2nd half dominance
    PRIORITY_TIERS = {
        'priority': ['SP2', 'N1', 'E0', 'E2', 'I1', 'T1', 'F1'],  # Strong 2nd half bias
        'moderate': ['G1', 'D2', 'P1', 'F2', 'E3', 'SC0', 'E1', 'B1'],
        'low_priority': ['D1', 'I2', 'SC1', 'SC2', 'SC3'],  # Weak 2nd half bias
    }

    def __init__(self, league_code='E0', league_half_ratio=None):
        """
        Initialize predictor

        Args:
            league_code: League identifier
            league_half_ratio: Override default ratio (2nd/1st half goals)
        """
        self.league_code = league_code

        if league_half_ratio is not None:
            self.league_ratio = league_half_ratio
        else:
            self.league_ratio = self.LEAGUE_HALF_RATIOS.get(league_code, 1.25)

        self.tier = self._get_tier(league_code)
        logger.info(f"Initialized HalfComparisonPredictor for {league_code} (tier: {self.tier}, ratio: {self.league_ratio:.3f})")

    def _get_tier(self, league_code):
        """Determine league priority tier for 2nd half market"""
        for tier, leagues in self.PRIORITY_TIERS.items():
            if league_code in leagues:
                return tier
        return 'moderate'

    def _poisson_prob(self, lam: float, k: int) -> float:
        """Calculate Poisson probability P(X=k)"""
        if lam < 0:
            return 0.0
        try:
            return (lam ** k) * math.exp(-lam) / math.factorial(k)
        except OverflowError:
            return 0.0

    def predict_half_scoring(self, xg_home: float, xg_away: float,
                            home_half_ratio: float = None, away_half_ratio: float = None,
                            is_youth_league: bool = False) -> Dict:
        """
        Predict 1st half vs 2nd half goal distribution

        Args:
            xg_home: Expected goals for home team (full match)
            xg_away: Expected goals for away team (full match)
            home_half_ratio: Home team's historical 2nd/1st half ratio
            away_half_ratio: Away team's historical 2nd/1st half ratio
            is_youth_league: Flag for youth/reserve teams (high fatigue)

        Returns:
            Dict with half probabilities and metadata
        """
        # Total expected goals
        xg_total = xg_home + xg_away

        # Determine effective ratio
        if home_half_ratio is None:
            home_half_ratio = self.league_ratio
        if away_half_ratio is None:
            away_half_ratio = self.league_ratio

        # Team-specific ratio (blend home and away patterns)
        team_ratio = (home_half_ratio + away_half_ratio) / 2

        # Youth league adjustment (extreme 2nd half dominance)
        if is_youth_league:
            team_ratio = max(team_ratio, 1.45)  # Force high ratio for youth

        # Blend team-specific and league-wide ratios
        # 60% team, 40% league (team patterns matter more for halves)
        final_ratio = 0.6 * team_ratio + 0.4 * self.league_ratio

        # Split xG between halves using ratio
        # If ratio = 1.3: 1st_half + 1.3*1st_half = total → 1st_half = total/2.3
        lambda_1st_total = xg_total / (1 + final_ratio)
        lambda_2nd_total = xg_total * final_ratio / (1 + final_ratio)

        # Calculate probability distribution for each half
        p_2nd_higher = 0.0
        p_1st_higher = 0.0
        p_equal = 0.0

        max_goals_per_half = 8

        for goals_1st in range(0, max_goals_per_half):
            for goals_2nd in range(0, max_goals_per_half):
                prob = (self._poisson_prob(lambda_1st_total, goals_1st) *
                       self._poisson_prob(lambda_2nd_total, goals_2nd))

                if goals_2nd > goals_1st:
                    p_2nd_higher += prob
                elif goals_1st > goals_2nd:
                    p_1st_higher += prob
                else:
                    p_equal += prob

        # Apply tier boost for priority leagues
        p_2nd_higher = self._apply_tier_boost(p_2nd_higher)

        # Renormalize after boost
        total = p_2nd_higher + p_1st_higher + p_equal
        p_2nd_higher /= total
        p_1st_higher /= total
        p_equal /= total

        # Confidence (how far from 33.3% baseline)
        confidence = max(
            abs(p_2nd_higher - 0.333),
            abs(p_1st_higher - 0.333),
            abs(p_equal - 0.333)
        ) * 1.5  # Scale to 0-1

        return {
            '2nd_Half': p_2nd_higher,
            '1st_Half': p_1st_higher,
            'Equal': p_equal,
            'confidence': confidence,
            'lambda_1st': lambda_1st_total,
            'lambda_2nd': lambda_2nd_total,
            'ratio_used': final_ratio,
            'team_ratio': team_ratio,
            'league_ratio': self.league_ratio,
            'tier': self.tier,
            'recommended': confidence >= 0.15 and p_2nd_higher >= 0.45,
        }

    def _apply_tier_boost(self, base_2nd_prob: float) -> float:
        """
        Apply league-tier boost to 2nd half probability

        Priority leagues (high 2nd half ratios) get boosted
        """
        tier_boosts = {
            'priority': 1.05,     # Boost 2nd half by 5%
            'moderate': 1.02,     # Small boost
            'low_priority': 0.98, # Slight penalty
        }

        boost = tier_boosts.get(self.tier, 1.0)

        # Apply boost to deviation from 0.333 baseline
        deviation = base_2nd_prob - 0.333
        boosted_deviation = deviation * boost
        adjusted_prob = 0.333 + boosted_deviation

        return max(0.01, min(0.99, adjusted_prob))

    def predict_with_game_state(self, xg_home: float, xg_away: float,
                                expected_ht_state: str = 'close',
                                **kwargs) -> Dict:
        """
        Adjust prediction based on expected half-time game state

        Args:
            xg_home, xg_away: Expected goals
            expected_ht_state: 'close' (0-0, 1-0), 'home_winning', 'away_winning',
                              'home_losing', 'away_losing'
            **kwargs: Additional args for predict_half_scoring

        Returns:
            Dict with adjusted probabilities
        """
        # Base prediction
        base_prediction = self.predict_half_scoring(xg_home, xg_away, **kwargs)

        # Game state adjustments (based on tactical psychology)
        state_adjustments = {
            'close': 1.0,           # 0-0 or 1-0 → normal 2nd half
            'home_winning': 0.92,   # Home ahead → may sit back
            'away_winning': 0.92,   # Away ahead → may sit back
            'home_losing': 1.12,    # Home behind → push forward (more 2nd half goals)
            'away_losing': 1.08,    # Away behind → push forward but less urgency
            'draw_expected': 1.05,  # Teams likely to be level → 2nd half push
        }

        adjustment = state_adjustments.get(expected_ht_state, 1.0)

        # Apply adjustment to 2nd half probability
        adjusted_2nd = base_prediction['2nd_Half'] * adjustment
        adjusted_1st = base_prediction['1st_Half'] * (2 - adjustment)  # Inverse effect

        # Renormalize
        total = adjusted_2nd + adjusted_1st + base_prediction['Equal']

        return {
            '2nd_Half': adjusted_2nd / total,
            '1st_Half': adjusted_1st / total,
            'Equal': base_prediction['Equal'] / total,
            'confidence': abs(adjusted_2nd / total - 0.333) * 1.5,
            'game_state_adjustment': adjustment,
            'base_2nd_prob': base_prediction['2nd_Half'],
            'tier': self.tier,
            'recommended': (adjusted_2nd / total) >= 0.45,
        }

    def predict_with_fatigue_model(self, xg_home: float, xg_away: float,
                                   home_days_rest: int = 7, away_days_rest: int = 7,
                                   home_recent_matches: int = 1, away_recent_matches: int = 1,
                                   **kwargs) -> Dict:
        """
        Predict with explicit fatigue modeling

        Fatigued teams have steeper 2nd half decline → more goals conceded

        Args:
            xg_home, xg_away: Expected goals
            home_days_rest, away_days_rest: Days since last match
            home_recent_matches, away_recent_matches: Matches in last 14 days
            **kwargs: Additional args

        Returns:
            Dict with fatigue-adjusted probabilities
        """
        # Base prediction
        base_prediction = self.predict_half_scoring(xg_home, xg_away, **kwargs)

        # Fatigue index (0 = fully rested, 1 = extremely fatigued)
        def calculate_fatigue(days_rest, recent_matches):
            rest_factor = max(0, (7 - days_rest) / 7)  # 0-1, higher = more fatigue
            load_factor = min(1, (recent_matches - 1) / 3)  # 0-1, more matches = fatigue
            return (rest_factor + load_factor) / 2

        home_fatigue = calculate_fatigue(home_days_rest, home_recent_matches)
        away_fatigue = calculate_fatigue(away_days_rest, away_recent_matches)
        avg_fatigue = (home_fatigue + away_fatigue) / 2

        # High fatigue → 2nd half dominance increases
        # Adjustment: +0% at no fatigue, up to +8% at extreme fatigue
        fatigue_boost = 1.0 + (avg_fatigue * 0.08)

        # Apply to 2nd half probability
        adjusted_2nd = base_prediction['2nd_Half'] * fatigue_boost

        # Renormalize
        total = adjusted_2nd + base_prediction['1st_Half'] + base_prediction['Equal']

        return {
            '2nd_Half': adjusted_2nd / total,
            '1st_Half': base_prediction['1st_Half'] / total,
            'Equal': base_prediction['Equal'] / total,
            'confidence': abs(adjusted_2nd / total - 0.333) * 1.5,
            'fatigue_boost': fatigue_boost,
            'avg_fatigue_index': avg_fatigue,
            'tier': self.tier,
            'recommended': (adjusted_2nd / total) >= 0.45,
        }

    def bulk_predict(self, matches: list) -> list:
        """
        Predict half comparison for multiple matches

        Args:
            matches: List of dicts with keys: xg_home, xg_away, home_half_ratio, etc.

        Returns:
            List of prediction dicts, sorted by confidence
        """
        predictions = []

        for match in matches:
            # Detect youth league
            is_youth = any(kw in f"{match.get('home_team', '')} {match.get('away_team', '')}".upper()
                          for kw in ['YOUTH', 'JONG', 'U23', 'U21', 'RESERVE', 'B TEAM'])

            pred = self.predict_half_scoring(
                match.get('xg_home', 1.5),
                match.get('xg_away', 1.5),
                match.get('home_half_ratio'),
                match.get('away_half_ratio'),
                is_youth_league=is_youth,
            )

            # Add match metadata
            pred['home_team'] = match.get('home_team', 'Unknown')
            pred['away_team'] = match.get('away_team', 'Unknown')
            pred['league'] = self.league_code
            pred['is_youth'] = is_youth

            predictions.append(pred)

        # Sort by confidence descending
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return predictions

    def get_league_summary(self) -> Dict:
        """Get summary statistics for the current league"""
        return {
            'league_code': self.league_code,
            'half_ratio': self.league_ratio,
            'tier': self.tier,
            'priority_score': self._calculate_priority_score(),
            'recommendation': self._get_recommendation(),
        }

    def _calculate_priority_score(self) -> float:
        """Calculate priority score for this league (0-100)"""
        # Based on empirical analysis
        ratio_deviation = self.league_ratio - 1.0  # Distance from neutral (1.0)

        tier_bonuses = {
            'priority': 20.0,
            'moderate': 10.0,
            'low_priority': 0.0,
        }

        score = (
            ratio_deviation * 200 +  # Higher ratio = higher score
            tier_bonuses.get(self.tier, 0)
        )

        return min(100, max(0, score))

    def _get_recommendation(self) -> str:
        """Get recommendation text for this league"""
        priority_score = self._calculate_priority_score()

        if priority_score >= 60:
            return "HIGHLY RECOMMENDED - Very strong 2nd half bias"
        elif priority_score >= 50:
            return "RECOMMENDED - Strong 2nd half bias"
        elif priority_score >= 40:
            return "CONSIDER - Moderate 2nd half bias"
        else:
            return "NOT RECOMMENDED - Weak 2nd half bias"


def compare_leagues(leagues: list = None) -> None:
    """
    Compare half-time characteristics across leagues

    Args:
        leagues: List of league codes to compare (default: priority leagues)
    """
    if leagues is None:
        leagues = ['SP2', 'N1', 'E0', 'E2', 'I1', 'F1', 'D2', 'D1', 'SC1']

    print("\n" + "="*90)
    print("HIGHEST SCORING HALF - LEAGUE COMPARISON")
    print("="*90)
    print(f"{'League':<8} {'Half Ratio':<12} {'Tier':<15} {'Priority':<10} {'Recommendation':<35}")
    print("-"*90)

    for league_code in leagues:
        predictor = HalfComparisonPredictor(league_code=league_code)
        summary = predictor.get_league_summary()

        print(f"{league_code:<8} {summary['half_ratio']:<12.3f} "
              f"{summary['tier']:<15} {summary['priority_score']:<10.1f} "
              f"{summary['recommendation']:<35}")

    print("="*90 + "\n")


if __name__ == '__main__':
    # Demonstration
    compare_leagues()

    # Example prediction for SP2 (La Liga 2) match
    print("\nExample: Real Oviedo vs Real Zaragoza (SP2 - La Liga 2)")
    print("-"*90)

    predictor = HalfComparisonPredictor(league_code='SP2')

    prediction = predictor.predict_half_scoring(
        xg_home=1.7,
        xg_away=1.4,
        home_half_ratio=1.40,  # Oviedo scores heavily in 2nd half
        away_half_ratio=1.35,  # Zaragoza also 2nd half team
    )

    print(f"2nd Half Probability: {prediction['2nd_Half']:.3f} ({prediction['2nd_Half']*100:.1f}%)")
    print(f"1st Half Probability: {prediction['1st_Half']:.3f} ({prediction['1st_Half']*100:.1f}%)")
    print(f"Equal Halves Probability: {prediction['Equal']:.3f} ({prediction['Equal']*100:.1f}%)")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"Recommended: {'YES' if prediction['recommended'] else 'NO'}")
    print(f"\nBreakdown:")
    print(f"  Expected 1st Half Goals: {prediction['lambda_1st']:.2f}")
    print(f"  Expected 2nd Half Goals: {prediction['lambda_2nd']:.2f}")
    print(f"  Effective Ratio: {prediction['ratio_used']:.3f}")
    print(f"  League Ratio (SP2): {prediction['league_ratio']:.3f}")

    # Fatigue-adjusted prediction
    print("\n\nFatigue-Adjusted Prediction (Both teams played 3 days ago):")
    print("-"*90)

    fatigue_pred = predictor.predict_with_fatigue_model(
        xg_home=1.7,
        xg_away=1.4,
        home_days_rest=3,
        away_days_rest=3,
        home_recent_matches=3,  # 3 matches in last 14 days
        away_recent_matches=3,
        home_half_ratio=1.40,
        away_half_ratio=1.35,
    )

    print(f"2nd Half Probability: {fatigue_pred['2nd_Half']:.3f} ({fatigue_pred['2nd_Half']*100:.1f}%)")
    print(f"Fatigue Boost Applied: {fatigue_pred['fatigue_boost']:.3f}x")
    print(f"Average Fatigue Index: {fatigue_pred['avg_fatigue_index']:.2f} (0=rested, 1=exhausted)")

