#!/usr/bin/env python3
"""
Real-World Parameter Impact Analyzer
====================================

This script analyzes actual analysis files to understand the real-world impact
of different parameter combinations on market selection and betting outcomes.

Author: Football Analytics AI Agent
Date: December 7, 2025
"""

import json
import glob
import os
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import statistics
from dataclasses import dataclass

@dataclass
class MarketAnalysis:
    """Analysis of market selection patterns"""
    market_type: str
    selection_count: int
    avg_probability: float
    avg_odds: float
    probability_range: Tuple[float, float]
    most_common_selections: List[Tuple[str, int]]

class RealWorldAnalyzer:
    """Analyzes real analysis files to understand parameter impacts"""

    def __init__(self, data_dir: str = "data/analysis"):
        self.data_dir = data_dir
        self.analysis_files = self._find_analysis_files()

    def _find_analysis_files(self) -> List[str]:
        """Find all analysis JSON files"""
        pattern = os.path.join(self.data_dir, "*suggestions*.json")
        files = glob.glob(pattern)
        return sorted(files)

    def analyze_market_patterns(self) -> Dict[str, Any]:
        """Analyze market selection patterns across all files"""
        market_stats = defaultdict(list)
        selection_patterns = defaultdict(Counter)
        probability_distributions = defaultdict(list)

        for file_path in self.analysis_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Handle both single objects and arrays
                suggestions = data.get('suggestions', [])
                if isinstance(data, list):
                    suggestions = data
                elif 'suggestions' not in data:
                    suggestions = [data]  # Single suggestion format

                for suggestion in suggestions:
                    picks = suggestion.get('picks', [])

                    for pick in picks:
                        market = pick.get('market', '')
                        selection = pick.get('selection', '')
                        prob = pick.get('prob', 0.0)
                        odds = pick.get('odds', 1.0)

                        market_stats[market].append({
                            'prob': prob,
                            'odds': odds,
                            'selection': selection
                        })

                        selection_patterns[market][selection] += 1
                        probability_distributions[market].append(prob)

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {file_path}: {e}")
                continue

        # Compile analysis
        analysis = {}
        for market in market_stats:
            picks = market_stats[market]
            probs = [p['prob'] for p in picks]
            odds = [p['odds'] for p in picks]

            analysis[market] = MarketAnalysis(
                market_type=market,
                selection_count=len(picks),
                avg_probability=statistics.mean(probs) if probs else 0,
                avg_odds=statistics.mean(odds) if odds else 0,
                probability_range=(min(probs), max(probs)) if probs else (0, 0),
                most_common_selections=selection_patterns[market].most_common(5)
            )

        return analysis

    def analyze_double_chance_patterns(self) -> Dict[str, Any]:
        """Specific analysis of Double Chance selections"""
        dc_data = []

        for file_path in self.analysis_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                suggestions = data.get('suggestions', [])
                if isinstance(data, list):
                    suggestions = data
                elif 'suggestions' not in data:
                    suggestions = [data]

                for suggestion in suggestions:
                    # Check both picks and markets
                    picks = suggestion.get('picks', [])
                    markets = suggestion.get('markets', {})
                    dc_markets = markets.get('DC', {})

                    # Analyze DC picks
                    for pick in picks:
                        if pick.get('market') == 'Double Chance':
                            dc_data.append({
                                'selection': pick.get('selection'),
                                'prob': pick.get('prob'),
                                'odds': pick.get('odds'),
                                'home': suggestion.get('home', ''),
                                'away': suggestion.get('away', ''),
                                'source': 'pick'
                            })

                    # Analyze available DC markets (even if not picked)
                    for dc_type, prob in dc_markets.items():
                        dc_data.append({
                            'selection': dc_type,
                            'prob': float(prob),
                            'odds': round(1.0/float(prob), 2) if float(prob) > 0 else 999,
                            'home': suggestion.get('home', ''),
                            'away': suggestion.get('away', ''),
                            'source': 'available'
                        })

            except Exception as e:
                continue

        # Analyze patterns
        if not dc_data:
            return {"error": "No Double Chance data found"}

        selection_counts = Counter([item['selection'] for item in dc_data])
        picked_items = [item for item in dc_data if item['source'] == 'pick']

        analysis = {
            'total_dc_opportunities': len(dc_data),
            'total_dc_picks': len(picked_items),
            'selection_rate': len(picked_items) / len(dc_data) * 100 if dc_data else 0,
            'selection_distribution': dict(selection_counts),
            'avg_probability': statistics.mean([item['prob'] for item in dc_data]),
            'avg_picked_probability': statistics.mean([item['prob'] for item in picked_items]) if picked_items else 0,
            'probability_thresholds': {
                '75th_percentile': sorted([item['prob'] for item in dc_data])[int(len(dc_data) * 0.75)] if dc_data else 0,
                '80th_percentile': sorted([item['prob'] for item in dc_data])[int(len(dc_data) * 0.80)] if dc_data else 0,
                '85th_percentile': sorted([item['prob'] for item in dc_data])[int(len(dc_data) * 0.85)] if dc_data else 0,
            }
        }

        return analysis

    def analyze_ml_impact(self) -> Dict[str, Any]:
        """Analyze ML prediction impact vs traditional models"""
        ml_comparisons = []

        for file_path in self.analysis_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                suggestions = data.get('suggestions', [])
                if isinstance(data, list):
                    suggestions = data
                elif 'suggestions' not in data:
                    suggestions = [data]

                for suggestion in suggestions:
                    ml_vs_poisson = suggestion.get('ml_vs_poisson', {})
                    ml_prediction = suggestion.get('ml_prediction', {})

                    if ml_vs_poisson and ml_prediction:
                        ml_comparisons.append({
                            'home': suggestion.get('home', ''),
                            'away': suggestion.get('away', ''),
                            'ml_vs_poisson': ml_vs_poisson,
                            'ml_prediction': ml_prediction
                        })

            except Exception as e:
                continue

        if not ml_comparisons:
            return {"error": "No ML comparison data found"}

        # Analyze deltas
        home_deltas = []
        draw_deltas = []
        away_deltas = []
        btts_deltas = []

        for comp in ml_comparisons:
            ml_vs_p = comp['ml_vs_poisson']
            if '1X2' in ml_vs_p:
                home_deltas.append(ml_vs_p['1X2'].get('delta_home', 0))
                draw_deltas.append(ml_vs_p['1X2'].get('delta_draw', 0))
                away_deltas.append(ml_vs_p['1X2'].get('delta_away', 0))

            if 'BTTS' in ml_vs_p:
                btts_deltas.append(ml_vs_p['BTTS'].get('delta_yes', 0))

        analysis = {
            'total_ml_comparisons': len(ml_comparisons),
            'average_deltas': {
                'home_win': statistics.mean(home_deltas) if home_deltas else 0,
                'draw': statistics.mean(draw_deltas) if draw_deltas else 0,
                'away_win': statistics.mean(away_deltas) if away_deltas else 0,
                'btts_yes': statistics.mean(btts_deltas) if btts_deltas else 0,
            },
            'significant_differences': {
                'large_home_favor': len([d for d in home_deltas if d > 0.1]),
                'large_away_favor': len([d for d in away_deltas if d > 0.1]),
                'large_btts_differences': len([d for d in btts_deltas if abs(d) > 0.1]),
            },
            'ml_model_usage': Counter([comp['ml_prediction'].get('model_1x2', 'Unknown') for comp in ml_comparisons])
        }

        return analysis

    def generate_parameter_recommendations(self) -> Dict[str, Any]:
        """Generate parameter recommendations based on real data analysis"""
        market_analysis = self.analyze_market_patterns()
        dc_analysis = self.analyze_double_chance_patterns()
        ml_analysis = self.analyze_ml_impact()

        recommendations = {
            'current_system_performance': {
                'most_selected_markets': sorted(
                    [(market, analysis.selection_count) for market, analysis in market_analysis.items()],
                    key=lambda x: x[1],
                    reverse=True
                ),
                'average_probabilities_by_market': {
                    market: analysis.avg_probability for market, analysis in market_analysis.items()
                }
            },

            'double_chance_insights': dc_analysis,
            'ml_impact_insights': ml_analysis,

            'recommended_configurations': {
                'based_on_current_data': {
                    'description': 'Optimized based on your actual selection patterns',
                    'observations': [
                        f"Double Chance selections show {dc_analysis.get('selection_rate', 0):.1f}% pick rate",
                        f"Average DC probability: {dc_analysis.get('avg_probability', 0):.3f}",
                        f"ML shows significant edge in {ml_analysis.get('significant_differences', {}).get('large_home_favor', 0)} home predictions"
                    ],
                    'suggested_params': self._generate_optimized_params(market_analysis, dc_analysis, ml_analysis)
                }
            }
        }

        return recommendations

    def _generate_optimized_params(self, market_analysis, dc_analysis, ml_analysis) -> Dict[str, Any]:
        """Generate optimized parameters based on analysis"""
        # Base recommendations on observed patterns
        avg_dc_prob = dc_analysis.get('avg_picked_probability', 0.75)

        # Adjust based on success patterns
        if avg_dc_prob > 0.8:
            dc_min = max(0.75, avg_dc_prob - 0.05)  # Slightly more aggressive
        else:
            dc_min = min(0.8, avg_dc_prob + 0.03)   # Slightly more conservative

        return {
            '--min-confidence': 0.65,  # Balanced based on typical patterns
            '--ml-mode': 'predict',     # Always use ML if available
            '--enable-double-chance': True,
            '--dc-min-prob': round(dc_min, 2),
            '--dc-secondary-threshold': round(dc_min + 0.05, 2),
            '--dc-allow-multiple': True,
            '--verbose': True
        }

    def print_comprehensive_analysis(self):
        """Print comprehensive analysis report"""
        print("=" * 80)
        print("REAL-WORLD PARAMETER IMPACT ANALYSIS")
        print("=" * 80)
        print(f"Analyzed {len(self.analysis_files)} analysis files")
        print()

        # Market patterns
        market_analysis = self.analyze_market_patterns()
        print("MARKET SELECTION PATTERNS:")
        print("-" * 40)
        for market, analysis in market_analysis.items():
            print(f"Market: {market}")
            print(f"  Total Selections: {analysis.selection_count}")
            print(f"  Avg Probability: {analysis.avg_probability:.3f}")
            print(f"  Avg Odds: {analysis.avg_odds:.2f}")
            print(f"  Probability Range: {analysis.probability_range[0]:.3f} - {analysis.probability_range[1]:.3f}")
            print(f"  Top Selections: {analysis.most_common_selections[:3]}")
            print()

        # Double Chance analysis
        dc_analysis = self.analyze_double_chance_patterns()
        if 'error' not in dc_analysis:
            print("DOUBLE CHANCE ANALYSIS:")
            print("-" * 40)
            print(f"Total DC Opportunities: {dc_analysis['total_dc_opportunities']}")
            print(f"Total DC Picks: {dc_analysis['total_dc_picks']}")
            print(f"Selection Rate: {dc_analysis['selection_rate']:.1f}%")
            print(f"Avg Probability (all): {dc_analysis['avg_probability']:.3f}")
            print(f"Avg Probability (picked): {dc_analysis['avg_picked_probability']:.3f}")
            print("Selection Distribution:", dc_analysis['selection_distribution'])
            print("Probability Thresholds:", dc_analysis['probability_thresholds'])
            print()

        # ML impact
        ml_analysis = self.analyze_ml_impact()
        if 'error' not in ml_analysis:
            print("MACHINE LEARNING IMPACT:")
            print("-" * 40)
            print(f"ML Comparisons Available: {ml_analysis['total_ml_comparisons']}")
            print("Average Deltas (ML vs Poisson):")
            for outcome, delta in ml_analysis['average_deltas'].items():
                print(f"  {outcome}: {delta:+.3f}")
            print("Significant Differences:", ml_analysis['significant_differences'])
            print()

        # Recommendations
        recommendations = self.generate_parameter_recommendations()
        print("OPTIMIZED PARAMETER RECOMMENDATIONS:")
        print("-" * 40)
        suggested = recommendations['recommended_configurations']['based_on_current_data']['suggested_params']
        for param, value in suggested.items():
            print(f"{param} {value}")
        print()
        print("Key Observations:")
        for obs in recommendations['recommended_configurations']['based_on_current_data']['observations']:
            print(f"- {obs}")

def main():
    """Run the real-world analysis"""
    analyzer = RealWorldAnalyzer()
    analyzer.print_comprehensive_analysis()

if __name__ == "__main__":
    main()
