"""
Odd/Even Goals Predictor with Lower-League Prioritization

Predicts whether total match goals will be Odd or Even, with:
- Poisson-based probability calculation
- Team historical bias integration
- League-specific prior adjustments
- LOWER DIVISION BOOSTING (E2, G1, SP2, SC1)

Empirical findings:
- English League One (E2): 54.07% odd rate
- Greek Super League (G1): 52.93% odd rate
- Scotland Championship (SC1): 52.70% odd rate
- vs Premier League (E0): 48.57% odd rate
"""

import math
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddEvenPredictor:
    """Predict odd/even total goals with league-aware boosting"""

    # Empirically derived league odd rates (from analysis)
    LEAGUE_ODD_RATES = {
        'E2': 0.5407,  # English League One - HIGHEST
        'G1': 0.5293,  # Greece Super League
        'SP2': 0.4923,  # Spain La Liga 2
        'N1': 0.4672,  # Netherlands Eredivisie
        'SC1': 0.5270,  # Scotland Championship
        'T1': 0.4807,  # Turkey Super Lig
        'F1': 0.4819,  # France Ligue 1
        'SC0': 0.4754,  # Scotland Premiership
        'E0': 0.4857,  # England Premier League (baseline)
        'I1': 0.4889,  # Italy Serie A
        'SP1': 0.5197,  # Spain La Liga
        'D2': 0.5047,  # Germany 2. Bundesliga
        'P1': 0.5130,  # Portugal Primeira Liga
        'F2': 0.5012,  # France Ligue 2
        'E3': 0.4913,  # England League Two
        'I2': 0.4670,  # Italy Serie B
        'SC3': 0.4483,  # Scotland League Two
        'D1': 0.4431,  # Germany Bundesliga - LOWEST
        'E1': 0.5120,  # England Championship
        'B1': 0.4934,  # Belgium Pro League
        'SC2': 0.5057,  # Scotland League One
    }

    # Priority tiers for boosting
    PRIORITY_TIERS = {
        'priority': ['E2', 'G1', 'SC1', 'SP2', 'E3'],  # Boost these heavily
        'moderate': ['N1', 'T1', 'F1', 'SC0', 'E1', 'SP1', 'D2', 'P1', 'F2'],
        'low_priority': ['E0', 'D1', 'I1', 'I2', 'B1'],  # Elite/stable leagues
    }

    def __init__(self, league_code='E0', league_prior_odd_rate=None):
        """
        Initialize predictor

        Args:
            league_code: League identifier (e.g., 'E2', 'G1')
            league_prior_odd_rate: Override default league odd rate
        """
        self.league_code = league_code

        if league_prior_odd_rate is not None:
            self.league_prior = league_prior_odd_rate
        else:
            self.league_prior = self.LEAGUE_ODD_RATES.get(league_code, 0.50)

        self.tier = self._get_tier(league_code)
        logger.info(f"Initialized OddEvenPredictor for {league_code} (tier: {self.tier}, prior: {self.league_prior:.3f})")

    def _get_tier(self, league_code):
        """Determine league priority tier"""
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

    def predict_from_poisson(self, xg_home: float, xg_away: float, max_goals: int = 10) -> Dict[str, float]:
        """
        Calculate odd/even probability using Poisson distribution

        Args:
            xg_home: Expected goals for home team
            xg_away: Expected goals for away team
            max_goals: Maximum goals to consider

        Returns:
            Dict with 'Odd' and 'Even' probabilities
        """
        p_odd = 0.0
        p_even = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                total = h + a
                prob = self._poisson_prob(xg_home, h) * self._poisson_prob(xg_away, a)

                if total % 2 == 1:
                    p_odd += prob
                else:
                    p_even += prob

        # Normalize
        total_prob = p_odd + p_even
        if total_prob > 0:
            p_odd /= total_prob
            p_even /= total_prob

        return {'Odd': p_odd, 'Even': p_even}

    def predict_with_team_bias(self, xg_home: float, xg_away: float,
                               home_odd_rate: float = None, away_odd_rate: float = None,
                               bayesian_weight: float = 0.25) -> Dict:
        """
        Combine Poisson prediction with team historical bias

        Args:
            xg_home: Expected goals home team
            xg_away: Expected goals away team
            home_odd_rate: Home team's historical odd match rate (0-1)
            away_odd_rate: Away team's historical odd match rate (0-1)
            bayesian_weight: Weight for team bias (0=pure Poisson, 1=pure team history)

        Returns:
            Dict with 'Odd', 'Even' probabilities and metadata
        """
        # Base Poisson prediction
        poisson_probs = self.predict_from_poisson(xg_home, xg_away)

        # If no team data provided, use league prior
        if home_odd_rate is None:
            home_odd_rate = self.league_prior
        if away_odd_rate is None:
            away_odd_rate = self.league_prior

        # Team average bias
        team_odd_rate = (home_odd_rate + away_odd_rate) / 2

        # Bayesian blend: (1-w)*Poisson + w*TeamBias
        final_odd_prob = (
            (1 - bayesian_weight) * poisson_probs['Odd'] +
            bayesian_weight * team_odd_rate
        )

        # Apply league-tier boost for lower divisions
        final_odd_prob = self._apply_tier_boost(final_odd_prob)

        # Confidence calculation
        confidence = abs(final_odd_prob - 0.5) * 2  # 0-1 scale

        return {
            'Odd': final_odd_prob,
            'Even': 1 - final_odd_prob,
            'confidence': confidence,
            'poisson_odd': poisson_probs['Odd'],
            'team_bias': team_odd_rate,
            'league_prior': self.league_prior,
            'tier': self.tier,
            'recommended': confidence >= 0.08,  # At least 54% probability
        }

    def _apply_tier_boost(self, base_odd_prob: float) -> float:
        """
        Apply league-tier boost to odd probability

        Lower divisions get boosted towards odd outcome
        Elite leagues get penalized slightly
        """
        # Tier-specific boost multipliers
        tier_boosts = {
            'priority': 1.08,     # Boost odd probability by 8%
            'moderate': 1.02,     # Slight boost
            'low_priority': 0.98, # Slight penalty (elite leagues more even)
        }

        boost = tier_boosts.get(self.tier, 1.0)

        # Apply boost to deviation from 0.5
        deviation = base_odd_prob - 0.5
        boosted_deviation = deviation * boost
        adjusted_prob = 0.5 + boosted_deviation

        # Clamp to valid probability range
        return max(0.01, min(0.99, adjusted_prob))

    def predict_with_full_context(self, xg_home: float, xg_away: float,
                                  home_odd_rate: float = None, away_odd_rate: float = None,
                                  home_recent_odd_streak: int = 0, away_recent_odd_streak: int = 0,
                                  h2h_odd_rate: float = None) -> Dict:
        """
        Full prediction with all available context

        Args:
            xg_home, xg_away: Expected goals
            home_odd_rate, away_odd_rate: Team season odd rates
            home_recent_odd_streak, away_recent_odd_streak: Recent match streak (negative for even)
            h2h_odd_rate: Head-to-head odd rate between these teams

        Returns:
            Dict with predictions and confidence metrics
        """
        # Base prediction with team bias
        base_prediction = self.predict_with_team_bias(xg_home, xg_away, home_odd_rate, away_odd_rate)

        # Streak adjustment (teams on odd/even streaks tend to continue short-term)
        streak_adjustment = 0.0
        if abs(home_recent_odd_streak) >= 3 or abs(away_recent_odd_streak) >= 3:
            avg_streak = (home_recent_odd_streak + away_recent_odd_streak) / 2
            streak_adjustment = np.sign(avg_streak) * min(0.03, abs(avg_streak) * 0.01)

        # H2H adjustment (if available and meaningful)
        h2h_adjustment = 0.0
        if h2h_odd_rate is not None and 0.3 <= h2h_odd_rate <= 0.7:
            h2h_adjustment = (h2h_odd_rate - 0.5) * 0.10  # Small weight for h2h

        # Low-scoring match bias (1-2 goals heavily favors odd)
        xg_total = xg_home + xg_away
        if xg_total < 2.3:
            # Low-scoring games: 1 goal = odd, 2 goals = even
            # But 1 goal is more common, so bias toward odd
            low_score_boost = (2.3 - xg_total) * 0.02
        else:
            low_score_boost = 0.0

        # Combined probability
        final_odd_prob = base_prediction['Odd'] + streak_adjustment + h2h_adjustment + low_score_boost
        final_odd_prob = max(0.01, min(0.99, final_odd_prob))

        # Enhanced confidence with context factors
        confidence = abs(final_odd_prob - 0.5) * 2

        # Boost confidence for priority leagues
        if self.tier == 'priority' and confidence >= 0.06:
            confidence = min(1.0, confidence * 1.15)

        return {
            'Odd': final_odd_prob,
            'Even': 1 - final_odd_prob,
            'confidence': confidence,
            'base_odd_prob': base_prediction['Odd'],
            'streak_adjustment': streak_adjustment,
            'h2h_adjustment': h2h_adjustment,
            'low_score_boost': low_score_boost,
            'tier': self.tier,
            'league_prior': self.league_prior,
            'recommended': confidence >= 0.08 and final_odd_prob >= 0.52,
            'xg_total': xg_total,
        }

    def bulk_predict(self, matches: list) -> list:
        """
        Predict odd/even for multiple matches

        Args:
            matches: List of dicts with keys: xg_home, xg_away, home_odd_rate, away_odd_rate

        Returns:
            List of prediction dicts, sorted by confidence
        """
        predictions = []

        for match in matches:
            pred = self.predict_with_full_context(
                match.get('xg_home', 1.5),
                match.get('xg_away', 1.5),
                match.get('home_odd_rate'),
                match.get('away_odd_rate'),
                match.get('home_recent_odd_streak', 0),
                match.get('away_recent_odd_streak', 0),
                match.get('h2h_odd_rate'),
            )

            # Add match metadata
            pred['home_team'] = match.get('home_team', 'Unknown')
            pred['away_team'] = match.get('away_team', 'Unknown')
            pred['league'] = self.league_code

            predictions.append(pred)

        # Sort by confidence descending
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return predictions

    def get_league_summary(self) -> Dict:
        """Get summary statistics for the current league"""
        return {
            'league_code': self.league_code,
            'league_odd_rate': self.league_prior,
            'tier': self.tier,
            'priority_score': self._calculate_priority_score(),
            'recommendation': self._get_recommendation(),
        }

    def _calculate_priority_score(self) -> float:
        """Calculate priority score for this league (0-100)"""
        # Based on empirical analysis
        odd_deviation = abs(self.league_prior - 0.5)

        tier_bonuses = {
            'priority': 20.0,
            'moderate': 10.0,
            'low_priority': 0.0,
        }

        score = (
            odd_deviation * 200 +  # Deviation from neutral
            tier_bonuses.get(self.tier, 0)
        )

        return min(100, score)

    def _get_recommendation(self) -> str:
        """Get recommendation text for this league"""
        priority_score = self._calculate_priority_score()

        if priority_score >= 55:
            return "HIGHLY RECOMMENDED - Strong odd/even signal"
        elif priority_score >= 45:
            return "RECOMMENDED - Moderate odd/even signal"
        elif priority_score >= 35:
            return "CONSIDER - Weak odd/even signal"
        else:
            return "NOT RECOMMENDED - Insufficient odd/even signal"


def compare_leagues(leagues: list = None) -> None:
    """
    Compare odd/even characteristics across leagues

    Args:
        leagues: List of league codes to compare (default: all priority leagues)
    """
    if leagues is None:
        leagues = ['E2', 'G1', 'SC1', 'SP2', 'N1', 'E0', 'D1', 'I1']

    print("\n" + "="*80)
    print("ODD/EVEN LEAGUE COMPARISON")
    print("="*80)
    print(f"{'League':<8} {'Odd Rate':<12} {'Tier':<15} {'Priority':<10} {'Recommendation':<30}")
    print("-"*80)

    for league_code in leagues:
        predictor = OddEvenPredictor(league_code=league_code)
        summary = predictor.get_league_summary()

        print(f"{league_code:<8} {summary['league_odd_rate']:<12.4f} "
              f"{summary['tier']:<15} {summary['priority_score']:<10.1f} "
              f"{summary['recommendation']:<30}")

    print("="*80 + "\n")


if __name__ == '__main__':
    # Demonstration
    compare_leagues()

    # Example prediction for E2 (League One) match
    print("\nExample: Portsmouth vs Oxford United (E2 - League One)")
    print("-"*80)

    predictor = OddEvenPredictor(league_code='E2')

    prediction = predictor.predict_with_full_context(
        xg_home=1.6,
        xg_away=1.3,
        home_odd_rate=0.58,  # Portsmouth historical odd rate
        away_odd_rate=0.51,  # Oxford historical odd rate
        home_recent_odd_streak=2,  # Won last 2 with odd goals
        away_recent_odd_streak=-1,  # Last match was even
    )

    print(f"Predicted Odd Probability: {prediction['Odd']:.3f} ({prediction['Odd']*100:.1f}%)")
    print(f"Predicted Even Probability: {prediction['Even']:.3f} ({prediction['Even']*100:.1f}%)")
    print(f"Confidence: {prediction['confidence']:.3f}")
    print(f"Recommended: {'YES' if prediction['recommended'] else 'NO'}")
    print(f"\nBreakdown:")
    print(f"  Base Poisson: {prediction['base_odd_prob']:.3f}")
    print(f"  Streak Adjustment: {prediction['streak_adjustment']:+.3f}")
    print(f"  Low-Score Boost: {prediction['low_score_boost']:+.3f}")
    print(f"  League Prior (E2): {prediction['league_prior']:.3f}")

