#!/usr/bin/env python3
"""
Football Analytics Parameter Optimization Suite
===============================================

Complete parameter analysis and optimization for multi-parlay strategies
based on real data analysis and theoretical considerations.

Author: Football Analytics AI Agent
Date: December 7, 2025
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ParamOptimization:
    """Parameter optimization configuration"""
    strategy_name: str
    description: str
    target_roi: str
    risk_level: str
    parameters: Dict[str, Any]
    expected_outcomes: Dict[str, Any]
    market_focus: List[str]

class FootballAnalyticsOptimizer:
    """Complete parameter optimization suite"""

    def __init__(self):
        self.current_analysis = {
            'dc_selection_rate': 26.1,
            'avg_dc_probability': 0.786,
            'ml_home_edge': 0.085,
            'ml_away_edge': -0.087,
            'most_selected_dc': '12',  # Draw or Win
            'avg_btts_prob': 0.692,
            'avg_over25_prob': 0.764
        }

    def get_optimization_strategies(self) -> Dict[str, ParamOptimization]:
        """Get optimized parameter strategies based on real data"""

        return {
            'conservative_multi_parlay': ParamOptimization(
                strategy_name='Conservative Multi-Parlay',
                description='High-probability selections for 5-8 leg parlays focusing on DC and safe markets',
                target_roi='15-25% per successful parlay',
                risk_level='Low (70-80% individual leg success rate)',
                parameters={
                    '--min-confidence': 0.72,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.82,
                    '--dc-secondary-threshold': 0.87,
                    '--dc-allow-multiple': False,
                    '--verbose': True
                },
                expected_outcomes={
                    'selections_per_day': '8-12 across all leagues',
                    'avg_odds_per_leg': '1.18-1.35',
                    'parlay_odds_5_legs': '2.4-4.5',
                    'parlay_odds_8_legs': '4.3-10.8',
                    'hit_rate_estimate': '12-18% (parlays)'
                },
                market_focus=['DC (primarily 12)', 'BTTS No', 'Under 3.5', 'Over 0.5']
            ),

            'balanced_value_hunter': ParamOptimization(
                strategy_name='Balanced Value Hunter',
                description='ML-enhanced value detection with mixed markets for 3-5 leg parlays',
                target_roi='25-40% per successful parlay',
                risk_level='Medium (60-75% individual leg success rate)',
                parameters={
                    '--min-confidence': 0.68,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.78,
                    '--dc-secondary-threshold': 0.83,
                    '--dc-allow-multiple': True,
                    '--verbose': True
                },
                expected_outcomes={
                    'selections_per_day': '12-18 across all leagues',
                    'avg_odds_per_leg': '1.25-1.65',
                    'parlay_odds_3_legs': '2.0-4.5',
                    'parlay_odds_5_legs': '3.1-11.4',
                    'hit_rate_estimate': '15-25% (parlays)'
                },
                market_focus=['DC (mixed)', 'BTTS Yes', 'Over 2.5', '1X2 (ML edge)', 'Over 1.5']
            ),

            'ml_edge_exploiter': ParamOptimization(
                strategy_name='ML Edge Exploiter',
                description='Maximum ML advantage exploitation for 2-4 leg parlays with higher variance',
                target_roi='40-80% per successful parlay',
                risk_level='Medium-High (55-70% individual leg success rate)',
                parameters={
                    '--min-confidence': 0.62,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.72,
                    '--dc-secondary-threshold': 0.78,
                    '--dc-allow-multiple': True,
                    '--verbose': True
                },
                expected_outcomes={
                    'selections_per_day': '15-25 across all leagues',
                    'avg_odds_per_leg': '1.35-2.2',
                    'parlay_odds_2_legs': '1.8-4.8',
                    'parlay_odds_4_legs': '3.3-23.4',
                    'hit_rate_estimate': '18-30% (parlays)'
                },
                market_focus=['1X2 (ML favored)', 'BTTS (ML edge)', 'Over/Under (ML adjusted)', 'DC (value spots)']
            ),

            'daily_accumulator': ParamOptimization(
                strategy_name='Daily Accumulator Builder',
                description='Daily 6-10 leg accumulators focusing on same-day fixtures with DC safety',
                target_roi='50-200% per successful accumulator',
                risk_level='High (individual legs), but safer market selection',
                parameters={
                    '--min-confidence': 0.75,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.85,
                    '--dc-secondary-threshold': 0.90,
                    '--dc-allow-multiple': False,
                    '--verbose': True
                },
                expected_outcomes={
                    'selections_per_day': '6-10 from day\'s fixtures',
                    'avg_odds_per_leg': '1.15-1.28',
                    'accumulator_odds_6_legs': '2.5-5.2',
                    'accumulator_odds_10_legs': '4.0-14.6',
                    'hit_rate_estimate': '8-15% (accumulators)'
                },
                market_focus=['DC (ultra-safe)', 'BTTS No (strong defenses)', 'Under 2.5 (tight games)']
            )
        }

    def get_market_specific_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations optimized for specific market types"""

        return {
            'double_chance_specialist': {
                'description': 'Maximizes Double Chance opportunities based on your 26.1% selection rate',
                'rationale': 'Your data shows DC avg prob of 0.786, suggesting current thresholds are working',
                'optimized_params': {
                    '--min-confidence': 0.60,  # Lower to catch more DC opportunities
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.75,     # Slightly lower than your 0.786 avg
                    '--dc-secondary-threshold': 0.82,
                    '--dc-allow-multiple': True,
                    '--verbose': True
                },
                'expected_improvement': '+15-20% more DC selections while maintaining quality'
            },

            'btts_value_hunter': {
                'description': 'Optimizes BTTS selection (your avg: 0.692 prob, 1.45 odds)',
                'rationale': 'Strong BTTS performance suggests room for more aggressive selection',
                'optimized_params': {
                    '--min-confidence': 0.65,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': False,  # Focus purely on BTTS
                    '--verbose': True
                },
                'expected_improvement': 'Better BTTS identification through ML edge detection'
            },

            'ml_home_advantage': {
                'description': 'Exploits ML +0.085 home edge vs Poisson models',
                'rationale': 'Your ML shows significant home team bias - 14 large home favors detected',
                'optimized_params': {
                    '--min-confidence': 0.58,  # Lower to catch ML edges
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.70,     # More aggressive for ML edges
                    '--dc-secondary-threshold': 0.75,
                    '--dc-allow-multiple': True,
                    '--verbose': True
                },
                'expected_improvement': 'Capture more home team value where ML beats traditional models'
            }
        }

    def generate_league_specific_commands(self) -> Dict[str, List[str]]:
        """Generate specific CLI commands for different leagues and strategies"""

        strategies = self.get_optimization_strategies()

        commands = {}
        leagues = ['E0', 'E1', 'D1', 'F1', 'I1', 'SP1', 'B1']

        for strategy_name, strategy in strategies.items():
            commands[strategy_name] = {}

            for league in leagues:
                params = []
                for param, value in strategy.parameters.items():
                    if isinstance(value, bool):
                        if value:
                            params.append(param)
                    else:
                        params.append(f"{param} {value}")

                command = f"python cli.py --task full-league --league {league} " + " ".join(params)
                commands[strategy_name][league] = command

        return commands

    def analyze_parlay_mathematics(self) -> Dict[str, Any]:
        """Analyze the mathematics behind parlay building with these parameters"""

        return {
            'probability_mathematics': {
                'conservative_parlay_5_legs': {
                    'individual_prob': 0.78,  # Based on your DC avg
                    'parlay_probability': 0.78 ** 5,
                    'parlay_percentage': f"{(0.78 ** 5) * 100:.1f}%",
                    'expected_odds': 1 / (0.78 ** 5),
                    'break_even_bookmaker_odds': f"{1 / (0.78 ** 5):.2f}"
                },
                'balanced_parlay_4_legs': {
                    'individual_prob': 0.70,
                    'parlay_probability': 0.70 ** 4,
                    'parlay_percentage': f"{(0.70 ** 4) * 100:.1f}%",
                    'expected_odds': 1 / (0.70 ** 4),
                    'break_even_bookmaker_odds': f"{1 / (0.70 ** 4):.2f}"
                },
                'aggressive_parlay_3_legs': {
                    'individual_prob': 0.65,
                    'parlay_probability': 0.65 ** 3,
                    'parlay_percentage': f"{(0.65 ** 3) * 100:.1f}%",
                    'expected_odds': 1 / (0.65 ** 3),
                    'break_even_bookmaker_odds': f"{1 / (0.65 ** 3):.2f}"
                }
            },

            'ml_edge_calculations': {
                'home_edge_value': f"+{self.current_analysis['ml_home_edge']:.3f}",
                'away_edge_value': f"{self.current_analysis['ml_away_edge']:.3f}",
                'potential_roi_boost': "5-15% when ML edges are properly identified and exploited",
                'edge_frequency': f"Significant edges in ~{(14+7)/33*100:.0f}% of ML comparisons"
            },

            'bankroll_management': {
                'conservative_strategy': '2-3% per parlay, higher volume',
                'balanced_strategy': '3-5% per parlay, medium volume',
                'aggressive_strategy': '5-8% per parlay, lower volume',
                'ml_edge_strategy': '4-6% when edge detected, otherwise 2-3%'
            }
        }

    def print_complete_optimization_guide(self):
        """Print the complete optimization guide"""

        print("=" * 100)
        print("FOOTBALL ANALYTICS PARAMETER OPTIMIZATION SUITE")
        print("=" * 100)
        print("Based on analysis of your actual data patterns")
        print()

        # Current system analysis
        print("CURRENT SYSTEM PERFORMANCE ANALYSIS:")
        print("-" * 50)
        print(f"Double Chance Selection Rate: {self.current_analysis['dc_selection_rate']:.1f}%")
        print(f"Average DC Probability: {self.current_analysis['avg_dc_probability']:.3f}")
        print(f"ML Home Team Edge: +{self.current_analysis['ml_home_edge']:.3f}")
        print(f"ML Away Team Edge: {self.current_analysis['ml_away_edge']:.3f}")
        print(f"Most Selected DC Type: {self.current_analysis['most_selected_dc']}")
        print()

        # Optimization strategies
        strategies = self.get_optimization_strategies()
        print("OPTIMIZED PARAMETER STRATEGIES:")
        print("-" * 50)

        for name, strategy in strategies.items():
            print(f"\nStrategy: {strategy.strategy_name}")
            print(f"Description: {strategy.description}")
            print(f"Risk Level: {strategy.risk_level}")
            print(f"Target ROI: {strategy.target_roi}")
            print("Parameters:")
            for param, value in strategy.parameters.items():
                print(f"  {param} {value}")
            print("Expected Outcomes:")
            for key, value in strategy.expected_outcomes.items():
                print(f"  {key}: {value}")
            print("Market Focus:", ", ".join(strategy.market_focus))
            print("-" * 80)

        # Market specific configs
        print("\nMARKET-SPECIFIC OPTIMIZATIONS:")
        print("-" * 50)

        market_configs = self.get_market_specific_configs()
        for config_name, config in market_configs.items():
            print(f"\nConfiguration: {config_name}")
            print(f"Description: {config['description']}")
            print(f"Rationale: {config['rationale']}")
            print("Optimized Parameters:")
            for param, value in config['optimized_params'].items():
                print(f"  {param} {value}")
            print(f"Expected Improvement: {config['expected_improvement']}")
            print()

        # Parlay mathematics
        print("PARLAY MATHEMATICS & EDGE ANALYSIS:")
        print("-" * 50)

        math_analysis = self.analyze_parlay_mathematics()

        print("Probability Mathematics:")
        for parlay_type, calc in math_analysis['probability_mathematics'].items():
            print(f"  {parlay_type}:")
            print(f"    Individual Leg Probability: {calc['individual_prob']}")
            print(f"    Parlay Success Rate: {calc['parlay_percentage']}")
            print(f"    Break-even Odds: {calc['break_even_bookmaker_odds']}")

        print("\nML Edge Analysis:")
        for key, value in math_analysis['ml_edge_calculations'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        print("\nBankroll Management:")
        for strategy, allocation in math_analysis['bankroll_management'].items():
            print(f"  {strategy.replace('_', ' ').title()}: {allocation}")

        # CLI commands
        print("\n" + "=" * 100)
        print("READY-TO-USE CLI COMMANDS:")
        print("=" * 100)

        commands = self.generate_league_specific_commands()

        print("\nCONSERVATIVE MULTI-PARLAY (Recommended for beginners):")
        for league, command in commands['conservative_multi_parlay'].items():
            print(f"{league}: {command}")

        print("\nBALANCED VALUE HUNTER (Recommended for experienced):")
        for league, command in commands['balanced_value_hunter'].items():
            print(f"{league}: {command}")

        print("\nML EDGE EXPLOITER (Advanced users):")
        for league, command in commands['ml_edge_exploiter'].items():
            print(f"{league}: {command}")

def main():
    """Run the complete optimization suite"""
    optimizer = FootballAnalyticsOptimizer()
    optimizer.print_complete_optimization_guide()

if __name__ == "__main__":
    main()
