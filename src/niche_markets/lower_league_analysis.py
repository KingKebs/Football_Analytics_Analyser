"""
Lower League Empirical Analysis

Analyzes football-data.co.uk CSV files to identify patterns in:
- Odd/Even goal distributions
- Half-time vs Full-time scoring patterns
- Goal volatility by league tier
- Late-goal frequency
- Defensive instability metrics

Focus: Lower divisions, reserves, and youth leagues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LowerLeagueAnalyzer:
    """Analyze lower league characteristics for niche markets"""

    # League tier classification
    LEAGUE_TIERS = {
        'elite': ['E0', 'D1', 'F1', 'I1', 'SP1', 'N1', 'P1', 'B1'],  # Top divisions
        'second_tier': ['E1', 'D2', 'F2', 'I2', 'SP2', 'SC0', 'SC1', 'G1', 'T1'],  # 2nd divisions
        'lower_division': ['E2', 'E3', 'SC2', 'SC3'],  # 3rd+ divisions
    }

    def __init__(self, data_dir: str = None):
        """Initialize analyzer with data directory"""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'football-data' / 'all-euro-football'
        self.data_dir = Path(data_dir)
        self.results = {}

    def load_league_data(self, league_code: str, seasons: List[str] = None) -> pd.DataFrame:
        """Load data for a specific league across seasons"""
        if seasons is None:
            seasons = ['2324', '2425', '2526']

        dfs = []
        for season in seasons:
            file_path = self.data_dir / f"{league_code}_{season}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    df['Season'] = season
                    df['League'] = league_code
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} matches from {league_code}_{season}")
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def calculate_half_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate half-time scoring metrics"""
        if df.empty or 'HTHG' not in df.columns:
            return {}

        # Filter valid matches and make a copy to avoid warnings
        df = df.dropna(subset=['FTHG', 'FTAG', 'HTHG', 'HTAG']).copy()

        # Calculate half goals
        df['1st_Half_Total'] = df['HTHG'] + df['HTAG']
        df['2nd_Half_Home'] = df['FTHG'] - df['HTHG']
        df['2nd_Half_Away'] = df['FTAG'] - df['HTAG']
        df['2nd_Half_Total'] = df['2nd_Half_Home'] + df['2nd_Half_Away']
        df['Total_Goals'] = df['FTHG'] + df['FTAG']

        # Highest scoring half
        df['2nd_Half_Higher'] = df['2nd_Half_Total'] > df['1st_Half_Total']
        df['1st_Half_Higher'] = df['1st_Half_Total'] > df['2nd_Half_Total']
        df['Halves_Equal'] = df['1st_Half_Total'] == df['2nd_Half_Total']

        # Calculate metrics
        metrics = {
            'avg_1st_half_goals': df['1st_Half_Total'].mean(),
            'avg_2nd_half_goals': df['2nd_Half_Total'].mean(),
            'half_ratio': df['2nd_Half_Total'].mean() / df['1st_Half_Total'].mean() if df['1st_Half_Total'].mean() > 0 else 1.0,
            '2nd_half_win_rate': df['2nd_Half_Higher'].mean(),
            '1st_half_win_rate': df['1st_Half_Higher'].mean(),
            'equal_halves_rate': df['Halves_Equal'].mean(),
            'std_1st_half': df['1st_Half_Total'].std(),
            'std_2nd_half': df['2nd_Half_Total'].std(),
        }

        return metrics

    def calculate_odd_even_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate odd/even goal distribution metrics"""
        if df.empty:
            return {}

        df = df.dropna(subset=['FTHG', 'FTAG']).copy()
        df['Total_Goals'] = df['FTHG'] + df['FTAG']
        df['Is_Odd'] = df['Total_Goals'] % 2 == 1

        metrics = {
            'odd_rate': df['Is_Odd'].mean(),
            'even_rate': (~df['Is_Odd']).mean(),
            'avg_total_goals': df['Total_Goals'].mean(),
            'std_total_goals': df['Total_Goals'].std(),
            'cv_total_goals': df['Total_Goals'].std() / df['Total_Goals'].mean() if df['Total_Goals'].mean() > 0 else 0,
        }

        # Distribution by goal count
        goal_dist = df['Total_Goals'].value_counts(normalize=True).sort_index()
        for goals in range(0, 7):
            metrics[f'p_exactly_{goals}_goals'] = goal_dist.get(goals, 0.0)

        return metrics

    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate goal volatility and defensive instability metrics"""
        if df.empty:
            return {}

        df = df.dropna(subset=['FTHG', 'FTAG', 'HTHG', 'HTAG']).copy()
        df['Total_Goals'] = df['FTHG'] + df['FTAG']
        df['2nd_Half_Total'] = (df['FTHG'] - df['HTHG']) + (df['FTAG'] - df['HTAG'])

        # Goal variance
        total_goals_variance = df['Total_Goals'].var()

        # High-scoring match frequency (4+ goals)
        high_scoring_rate = (df['Total_Goals'] >= 4).mean()

        # Very high-scoring (5+ goals)
        very_high_scoring_rate = (df['Total_Goals'] >= 5).mean()

        # Goal swing (large 2nd half changes)
        df['1st_Half_Total'] = df['HTHG'] + df['HTAG']
        df['Half_Swing'] = abs(df['2nd_Half_Total'] - df['1st_Half_Total'])
        avg_swing = df['Half_Swing'].mean()

        # Late goals (using 2nd half as proxy for "late")
        late_goal_intensity = df['2nd_Half_Total'].mean()

        # Clean sheet failure rate (both teams score)
        both_score_rate = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).mean()

        # Comeback frequency (team losing at HT wins at FT)
        df['Home_Comeback'] = (df['HTHG'] < df['HTAG']) & (df['FTHG'] > df['FTAG'])
        df['Away_Comeback'] = (df['HTAG'] < df['HTHG']) & (df['FTAG'] > df['FTHG'])
        comeback_rate = (df['Home_Comeback'] | df['Away_Comeback']).mean()

        # Lead changes (HT leader != FT leader)
        df['HT_Leader'] = np.where(df['HTHG'] > df['HTAG'], 'H',
                                    np.where(df['HTAG'] > df['HTHG'], 'A', 'D'))
        df['FT_Leader'] = np.where(df['FTHG'] > df['FTAG'], 'H',
                                    np.where(df['FTAG'] > df['FTHG'], 'A', 'D'))
        lead_change_rate = (df['HT_Leader'] != df['FT_Leader']).mean()

        metrics = {
            'total_goals_variance': total_goals_variance,
            'high_scoring_rate': high_scoring_rate,
            'very_high_scoring_rate': very_high_scoring_rate,
            'avg_half_swing': avg_swing,
            'late_goal_intensity': late_goal_intensity,
            'both_score_rate': both_score_rate,
            'comeback_rate': comeback_rate,
            'lead_change_rate': lead_change_rate,
        }

        return metrics

    def calculate_tactical_inconsistency(self, df: pd.DataFrame) -> Dict:
        """Calculate metrics indicating tactical inconsistency"""
        if df.empty or 'HomeTeam' not in df.columns:
            return {}

        df = df.dropna(subset=['FTHG', 'FTAG', 'HomeTeam', 'AwayTeam']).copy()

        # Team-level goal variance
        team_goal_variance = []
        for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
            home_goals = df[df['HomeTeam'] == team]['FTHG']
            away_goals = df[df['AwayTeam'] == team]['FTAG']
            all_goals = pd.concat([home_goals, away_goals])
            if len(all_goals) > 0:
                team_goal_variance.append(all_goals.std())

        avg_team_variance = np.mean(team_goal_variance) if team_goal_variance else 0

        # Result unpredictability (home win rate close to 33.3%)
        home_win_rate = (df['FTHG'] > df['FTAG']).mean()
        draw_rate = (df['FTHG'] == df['FTAG']).mean()
        away_win_rate = (df['FTAG'] > df['FTHG']).mean()

        # Entropy of results (higher = more unpredictable)
        result_probs = [home_win_rate, draw_rate, away_win_rate]
        result_entropy = -sum(p * np.log(p) if p > 0 else 0 for p in result_probs)

        metrics = {
            'avg_team_goal_variance': avg_team_variance,
            'home_win_rate': home_win_rate,
            'draw_rate': draw_rate,
            'away_win_rate': away_win_rate,
            'result_entropy': result_entropy,
        }

        return metrics

    def analyze_league(self, league_code: str) -> Dict:
        """Complete analysis for a single league"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {league_code}")
        logger.info(f"{'='*60}")

        df = self.load_league_data(league_code)

        if df.empty:
            logger.warning(f"No data found for {league_code}")
            return {}

        results = {
            'league_code': league_code,
            'total_matches': len(df),
            'half_metrics': self.calculate_half_metrics(df),
            'odd_even_metrics': self.calculate_odd_even_metrics(df),
            'volatility_metrics': self.calculate_volatility_metrics(df),
            'tactical_metrics': self.calculate_tactical_inconsistency(df),
        }

        # Calculate composite volatility score
        vol = results['volatility_metrics']
        results['volatility_score'] = (
            vol.get('total_goals_variance', 0) * 0.3 +
            vol.get('high_scoring_rate', 0) * 100 * 0.2 +
            vol.get('avg_half_swing', 0) * 10 * 0.2 +
            vol.get('lead_change_rate', 0) * 100 * 0.3
        )

        return results

    def compare_tiers(self) -> Dict:
        """Compare elite vs lower division leagues"""
        all_results = {}

        for tier_name, leagues in self.LEAGUE_TIERS.items():
            logger.info(f"\n{'#'*60}")
            logger.info(f"Processing {tier_name.upper()} tier")
            logger.info(f"{'#'*60}")

            tier_results = []
            for league in leagues:
                result = self.analyze_league(league)
                if result:
                    result['tier'] = tier_name
                    tier_results.append(result)

            all_results[tier_name] = tier_results

        return all_results

    def generate_priority_scores(self, all_results: Dict) -> pd.DataFrame:
        """Generate league priority scores for niche markets"""
        rows = []

        for tier_name, tier_results in all_results.items():
            for result in tier_results:
                league = result['league_code']

                # Extract key metrics
                half_metrics = result.get('half_metrics', {})
                odd_metrics = result.get('odd_even_metrics', {})
                vol_metrics = result.get('volatility_metrics', {})

                # Calculate priority scores

                # 1. Odd/Even Priority Score
                odd_rate = odd_metrics.get('odd_rate', 0.5)
                odd_deviation = abs(odd_rate - 0.5)  # Distance from neutral
                goal_cv = odd_metrics.get('cv_total_goals', 0)

                odd_even_score = (
                    odd_deviation * 200 +  # Higher deviation = better
                    goal_cv * 50 +         # Higher variance = better
                    (odd_rate > 0.52) * 20  # Bonus if odd-biased
                )

                # 2. 2nd Half Priority Score
                half_ratio = half_metrics.get('half_ratio', 1.0)
                second_half_win = half_metrics.get('2nd_half_win_rate', 0.33)
                half_std_diff = half_metrics.get('std_2nd_half', 0) - half_metrics.get('std_1st_half', 0)

                second_half_score = (
                    (half_ratio - 1.0) * 100 +    # Ratio above 1.0
                    second_half_win * 100 +        # Win rate
                    max(0, half_std_diff) * 20     # Higher 2nd half variance
                )

                # 3. Overall Volatility Score
                volatility_score = result.get('volatility_score', 0)

                # 4. Combined Priority Score (weighted)
                combined_score = (
                    odd_even_score * 0.4 +
                    second_half_score * 0.4 +
                    volatility_score * 0.2
                )

                rows.append({
                    'league': league,
                    'tier': tier_name,
                    'matches': result['total_matches'],
                    'odd_rate': odd_rate,
                    'half_ratio': half_ratio,
                    '2nd_half_win_rate': second_half_win,
                    'volatility': volatility_score,
                    'odd_even_priority': odd_even_score,
                    '2nd_half_priority': second_half_score,
                    'combined_priority': combined_score,
                })

        df = pd.DataFrame(rows)
        df = df.sort_values('combined_priority', ascending=False)

        return df

    def save_results(self, output_dir: str = None):
        """Save analysis results"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / 'data' / 'analysis'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze all tiers
        all_results = self.compare_tiers()

        # Generate priority scores
        priority_df = self.generate_priority_scores(all_results)

        # Save to CSV
        priority_csv = output_dir / 'league_priority_scores.csv'
        priority_df.to_csv(priority_csv, index=False)
        logger.info(f"\nSaved priority scores to {priority_csv}")

        # Save detailed JSON
        results_json = output_dir / 'lower_league_analysis.json'
        with open(results_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Saved detailed analysis to {results_json}")

        # Print summary
        self.print_summary(priority_df)

        return priority_df, all_results

    def print_summary(self, priority_df: pd.DataFrame):
        """Print analysis summary"""
        logger.info("\n" + "="*80)
        logger.info("LEAGUE PRIORITY RANKINGS FOR NICHE MARKETS")
        logger.info("="*80)

        logger.info("\nTOP 10 LEAGUES FOR ODD/EVEN MARKET:")
        logger.info("-" * 80)
        top_odd = priority_df.nlargest(10, 'odd_even_priority')[
            ['league', 'tier', 'odd_rate', 'odd_even_priority']
        ]
        logger.info(top_odd.to_string(index=False))

        logger.info("\n\nTOP 10 LEAGUES FOR 2ND HALF MARKET:")
        logger.info("-" * 80)
        top_half = priority_df.nlargest(10, '2nd_half_priority')[
            ['league', 'tier', 'half_ratio', '2nd_half_win_rate', '2nd_half_priority']
        ]
        logger.info(top_half.to_string(index=False))

        logger.info("\n\nTOP 10 COMBINED PRIORITY:")
        logger.info("-" * 80)
        top_combined = priority_df.nlargest(10, 'combined_priority')[
            ['league', 'tier', 'odd_rate', 'half_ratio', 'combined_priority']
        ]
        logger.info(top_combined.to_string(index=False))

        # Tier averages
        logger.info("\n\nAVERAGE METRICS BY TIER:")
        logger.info("-" * 80)
        tier_avg = priority_df.groupby('tier').agg({
            'odd_rate': 'mean',
            'half_ratio': 'mean',
            '2nd_half_win_rate': 'mean',
            'volatility': 'mean',
            'combined_priority': 'mean',
        }).round(3)
        logger.info(tier_avg.to_string())


def main():
    """Run lower league analysis"""
    analyzer = LowerLeagueAnalyzer()
    priority_df, all_results = analyzer.save_results()

    return priority_df, all_results


if __name__ == '__main__':
    main()

