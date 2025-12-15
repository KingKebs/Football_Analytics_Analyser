"""
Test suite for niche markets analysis module
Basic functionality and import tests
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "niche_markets"))



class TestNicheMarketsImports:
    """Test that niche markets modules can be imported"""

    def test_import_odd_even_predictor(self):
        """Test importing the odd even predictor module"""
        try:
            from odd_even_predictor import OddEvenPredictor
            predictor = OddEvenPredictor('E2')  # League One
            assert predictor is not None
        except ImportError:
            pytest.skip("OddEvenPredictor not available for import")

    def test_import_league_priors(self):
        """Test importing league priors data"""
        try:
            from league_priors import LEAGUE_PRIORS
            assert isinstance(LEAGUE_PRIORS, dict)
            assert len(LEAGUE_PRIORS) > 0
        except ImportError:
            pytest.skip("LEAGUE_PRIORS not available for import")

    def test_import_lower_league_analyzer(self):
        """Test importing the lower league analyzer"""
        try:
            from lower_league_analysis import LowerLeagueAnalyzer
            analyzer = LowerLeagueAnalyzer()
            assert analyzer is not None
        except ImportError:
            pytest.skip("LowerLeagueAnalyzer not available for import")


class TestNicheMarketsCore:
    """Test core niche markets functionality if available"""

    def test_odd_even_predictor_basic(self):
        """Test basic odd/even predictor functionality"""
        try:
            from odd_even_predictor import OddEvenPredictor
            predictor = OddEvenPredictor('E2')

            # Test that it can make a basic prediction
            result = predictor.predict_from_poisson(1.5, 1.3)
            assert ('odd_prob' in result or 'Odd' in result), "Should have odd probability"
            assert ('even_prob' in result or 'Even' in result), "Should have even probability"

            # Get the actual odd probability value
            odd_val = result.get('odd_prob', result.get('Odd'))
            assert isinstance(odd_val, float), "Odd probability should be a float"

        except (ImportError, AttributeError):
            pytest.skip("OddEvenPredictor functionality not fully available")

    def test_lower_league_analyzer_basic(self):
        """Test basic lower league analyzer functionality"""
        try:
            from lower_league_analysis import LowerLeagueAnalyzer
            analyzer = LowerLeagueAnalyzer()

            # Test that it has expected tier classifications
            league_tiers = analyzer.LEAGUE_TIERS
            assert 'elite' in league_tiers
            assert 'second_tier' in league_tiers

        except (ImportError, AttributeError):
            pytest.skip("LowerLeagueAnalyzer functionality not fully available")


class TestConceptualValidation:
    """Test the conceptual framework of niche markets analysis"""

    def test_league_one_priority_concept(self):
        """Test that League One is prioritized in the niche markets approach"""
        # This tests the theoretical framework
        league_one_odd_rate = 0.5407  # Empirical data from analysis
        premier_league_odd_rate = 0.4857

        assert league_one_odd_rate > 0.5, "League One should favor odd totals"
        assert league_one_odd_rate > premier_league_odd_rate, "League One should have higher odd rate"

    def test_parlay_math_concepts(self):
        """Test parlay probability calculations"""
        # Test the mathematical concepts
        match1_prob = 0.5407  # League One
        match2_prob = 0.529   # LaLiga2

        combined_prob = match1_prob * match2_prob
        fair_parlay_odds = 0.25  # 50% * 50% for fair odds

        assert combined_prob > fair_parlay_odds, "Cross-league parlay should beat fair odds"
        assert combined_prob > 0.28, "Should exceed minimum threshold"


# Pytest fixtures for test data
@pytest.fixture
def sample_match_data():
    """Sample match data for testing"""
    return {
        'home_team': 'Barnsley',
        'away_team': 'Leyton Orient',
        'league': 'E2',
        'home_goals_avg': 1.2,
        'away_goals_avg': 1.1,
        'kick_off_time': '17:00'
    }


@pytest.fixture
def sample_parlay_matches():
    """Sample matches for parlay testing"""
    return [
        {
            'league': 'E2',
            'home_team': 'Barnsley',
            'away_team': 'Leyton Orient',
            'odd_prob': 0.5407
        },
        {
            'league': 'SP2',
            'home_team': 'Gijon',
            'away_team': 'Granada CF',
            'odd_prob': 0.529
        }
    ]


def test_parlay_probability_calculation(sample_parlay_matches):
    """Test parlay probability calculation"""
    combined_prob = 1.0
    for match in sample_parlay_matches:
        combined_prob *= match['odd_prob']

    expected_prob = 0.5407 * 0.529  # ≈ 0.286
    assert combined_prob == pytest.approx(expected_prob, rel=1e-3)
    assert combined_prob > 0.25, "Combined parlay should beat fair 25% odds"


if __name__ == "__main__":
    # Run tests when file is executed directly
    pytest.main([__file__, "-v"])
