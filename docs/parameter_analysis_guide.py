#!/usr/bin/env python3
"""
Football Analytics Parameter Analysis Guide
==========================================

This script documents and analyzes the CLI parameters used in the Football Analytics system,
their impact on different betting markets, and optimal combinations for multi-parlay strategies.

Author: Football Analytics AI Agent
Date: December 7, 2025
"""

import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class MarketType(Enum):
    """Betting market types supported by the system"""
    ONE_X_TWO = "1X2"
    DOUBLE_CHANCE = "DC"
    OVER_UNDER = "OU"
    BTTS = "BTTS"
    CORNERS = "Corners"
    CARDS = "Cards"

class ConfidenceLevel(Enum):
    """Confidence level categories"""
    LOW = "Low (0.5-0.65)"
    MEDIUM = "Medium (0.65-0.75)"
    HIGH = "High (0.75-0.85)"
    VERY_HIGH = "Very High (0.85+)"

@dataclass
class ParameterConfig:
    """Configuration for a specific parameter"""
    name: str
    description: str
    data_type: str
    default_value: Any
    valid_range: str
    market_impact: List[MarketType]
    strategic_purpose: str
    ml_enhancement: str
    parlay_considerations: str

class FootballAnalyticsParameterGuide:
    """
    Comprehensive guide to Football Analytics CLI parameters
    """

    def __init__(self):
        self.parameters = self._initialize_parameters()
        self.market_combinations = self._initialize_market_combinations()
        self.ml_advantages = self._initialize_ml_advantages()

    def _initialize_parameters(self) -> Dict[str, ParameterConfig]:
        """Initialize all parameter configurations"""
        return {
            'min_confidence': ParameterConfig(
                name='--min-confidence',
                description='Minimum probability threshold for surfacing market suggestions',
                data_type='float',
                default_value=0.6,
                valid_range='0.1 to 1.0 (recommended: 0.5-0.8)',
                market_impact=[MarketType.ONE_X_TWO, MarketType.DOUBLE_CHANCE, MarketType.OVER_UNDER, MarketType.BTTS],
                strategic_purpose='Quality filter - Higher values = fewer but more confident picks. Lower values = more picks but potentially lower quality.',
                ml_enhancement='ML predictions can identify value at lower confidence thresholds where traditional models might miss opportunities.',
                parlay_considerations='For multi-parlay: Use 0.6-0.7 to balance quantity and quality. Higher for safer accumulators.'
            ),

            'ml_mode': ParameterConfig(
                name='--ml-mode',
                description='Machine Learning integration mode',
                data_type='choice',
                default_value='off',
                valid_range='off | train | predict',
                market_impact=[MarketType.ONE_X_TWO, MarketType.BTTS, MarketType.OVER_UNDER],
                strategic_purpose='Enhances traditional Poisson/XG models with ML predictions for goals markets.',
                ml_enhancement='Provides alternative probability estimates that can reveal value where bookmakers misprice markets.',
                parlay_considerations='Essential for multi-parlay strategy - ML can identify undervalued selections across different matches.'
            ),

            'enable_double_chance': ParameterConfig(
                name='--enable-double-chance',
                description='Enable Double Chance market analysis (1X, X2, 12)',
                data_type='boolean',
                default_value=False,
                valid_range='True | False',
                market_impact=[MarketType.DOUBLE_CHANCE],
                strategic_purpose='Provides safer betting options by covering two outcomes in one bet.',
                ml_enhancement='ML can better assess when Double Chance offers genuine value vs straight 1X2.',
                parlay_considerations='Critical for parlay building - DC selections reduce risk while maintaining decent odds.'
            ),

            'dc_min_prob': ParameterConfig(
                name='--dc-min-prob',
                description='Minimum probability threshold for Double Chance selection',
                data_type='float',
                default_value=0.75,
                valid_range='0.5 to 0.95 (recommended: 0.7-0.85)',
                market_impact=[MarketType.DOUBLE_CHANCE],
                strategic_purpose='Quality control for DC picks - ensures only high-probability DC selections surface.',
                ml_enhancement='ML can identify DC value at different probability levels based on historical performance.',
                parlay_considerations='Key parlay parameter - 0.75-0.8 balances safety with meaningful odds contribution.'
            ),

            'dc_secondary_threshold': ParameterConfig(
                name='--dc-secondary-threshold',
                description='Secondary threshold for adding DC alongside strong 1X2 picks',
                data_type='float',
                default_value=0.80,
                valid_range='0.6 to 0.95 (recommended: 0.75-0.9)',
                market_impact=[MarketType.DOUBLE_CHANCE, MarketType.ONE_X_TWO],
                strategic_purpose='Allows inclusion of very high-probability DC picks even when strong 1X2 exists.',
                ml_enhancement='ML can optimize when to prefer DC vs 1X2 based on edge detection.',
                parlay_considerations='Enables dual-market strategy - both safer DC and higher-odds 1X2 in different legs.'
            ),

            'dc_allow_multiple': ParameterConfig(
                name='--dc-allow-multiple',
                description='Allow multiple Double Chance selections per match',
                data_type='boolean',
                default_value=False,
                valid_range='True | False',
                market_impact=[MarketType.DOUBLE_CHANCE],
                strategic_purpose='Permits both 1X2 and DC selections from same match when both meet thresholds.',
                ml_enhancement='ML can determine optimal market mix for maximum expected value.',
                parlay_considerations='Increases selection pool for parlay construction but requires careful odds management.'
            ),

            'verbose': ParameterConfig(
                name='--verbose',
                description='Enable detailed output and logging',
                data_type='boolean',
                default_value=False,
                valid_range='True | False',
                market_impact=[],
                strategic_purpose='Provides detailed analysis for manual review and strategy refinement.',
                ml_enhancement='Shows ML vs traditional model comparisons for informed decision making.',
                parlay_considerations='Essential for understanding why specific combinations were selected.'
            )
        }

    def _initialize_market_combinations(self) -> Dict[str, Dict]:
        """Define optimal parameter combinations for different market strategies"""
        return {
            'conservative_parlay': {
                'description': 'Safe multi-parlay with high-probability selections',
                'target_markets': ['DC', 'BTTS_No', 'Under2.5'],
                'parameters': {
                    '--min-confidence': 0.75,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.80,
                    '--dc-secondary-threshold': 0.85,
                    '--dc-allow-multiple': False
                },
                'expected_odds_range': '1.2-1.6 per selection',
                'risk_level': 'Low',
                'typical_parlay_size': '4-6 selections'
            },

            'balanced_parlay': {
                'description': 'Balanced risk/reward multi-parlay strategy',
                'target_markets': ['DC', '1X2', 'BTTS', 'Over1.5'],
                'parameters': {
                    '--min-confidence': 0.65,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.75,
                    '--dc-secondary-threshold': 0.80,
                    '--dc-allow-multiple': True
                },
                'expected_odds_range': '1.3-2.0 per selection',
                'risk_level': 'Medium',
                'typical_parlay_size': '3-5 selections'
            },

            'value_hunter': {
                'description': 'ML-enhanced value detection across markets',
                'target_markets': ['1X2', 'BTTS', 'Over2.5', 'DC'],
                'parameters': {
                    '--min-confidence': 0.60,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.70,
                    '--dc-secondary-threshold': 0.75,
                    '--dc-allow-multiple': True
                },
                'expected_odds_range': '1.4-2.5 per selection',
                'risk_level': 'Medium-High',
                'typical_parlay_size': '3-4 selections'
            },

            'ml_edge_strategy': {
                'description': 'Maximum ML advantage exploitation',
                'target_markets': ['All markets where ML shows significant edge'],
                'parameters': {
                    '--min-confidence': 0.55,
                    '--ml-mode': 'predict',
                    '--enable-double-chance': True,
                    '--dc-min-prob': 0.65,
                    '--dc-secondary-threshold': 0.70,
                    '--dc-allow-multiple': True
                },
                'expected_odds_range': '1.5-3.0 per selection',
                'risk_level': 'High',
                'typical_parlay_size': '2-3 selections'
            }
        }

    def _initialize_ml_advantages(self) -> Dict[str, str]:
        """Document how ML enhances each market type"""
        return {
            '1X2_markets': '''
                ML Enhancement for 1X2 Markets:
                - Captures non-linear relationships between team form, fixtures, and outcomes
                - Identifies value when traditional Poisson models over/underestimate probabilities
                - Learns from recent performance patterns that XG models might miss
                - Can detect fixture congestion, motivation, and tactical matchup impacts
            ''',

            'double_chance_markets': '''
                ML Enhancement for Double Chance Markets:
                - Better assessment of when DC offers genuine value vs inflated confidence
                - Identifies matches where variance is higher than traditional models suggest
                - Learns optimal thresholds for DC selection based on historical ROI
                - Can detect when bookmakers misprice DC relative to constituent outcomes
            ''',

            'btts_markets': '''
                ML Enhancement for BTTS Markets:
                - Captures defensive/offensive balance beyond simple XG metrics
                - Identifies teams' propensity for high/low-scoring games in specific contexts
                - Learns from tactical setups and their impact on goal-scoring patterns
                - Better handling of clean sheet specialists vs porous defenses
            ''',

            'over_under_markets': '''
                ML Enhancement for Over/Under Markets:
                - More sophisticated total goals prediction than Poisson distribution
                - Learns from match circumstances (weather, stakes, timing) affecting goals
                - Identifies when traditional models systematically over/underestimate totals
                - Better correlation modeling between team XG and actual goal variance
            '''
        }

    def analyze_current_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a given parameter configuration"""
        analysis = {
            'configuration': config,
            'market_focus': [],
            'risk_assessment': '',
            'parlay_suitability': '',
            'ml_utilization': '',
            'recommendations': []
        }

        # Analyze min_confidence impact
        min_conf = config.get('min_confidence', 0.6)
        if min_conf >= 0.75:
            analysis['risk_assessment'] = 'Conservative - High quality, fewer picks'
            analysis['parlay_suitability'] = 'Excellent for larger parlays (5+ selections)'
        elif min_conf >= 0.65:
            analysis['risk_assessment'] = 'Balanced - Good quality/quantity ratio'
            analysis['parlay_suitability'] = 'Good for medium parlays (3-5 selections)'
        else:
            analysis['risk_assessment'] = 'Aggressive - More picks, higher variance'
            analysis['parlay_suitability'] = 'Best for smaller parlays (2-3 selections)'

        # Analyze ML mode
        ml_mode = config.get('ml_mode', 'off')
        if ml_mode == 'predict':
            analysis['ml_utilization'] = 'Full ML integration - Enhanced edge detection'
            analysis['market_focus'].extend(['Goals markets with ML edge'])
        elif ml_mode == 'train':
            analysis['ml_utilization'] = 'Model training mode - Building predictive capability'
        else:
            analysis['ml_utilization'] = 'Traditional models only - Missing ML advantages'
            analysis['recommendations'].append('Enable ML mode for enhanced predictions')

        # Analyze Double Chance configuration
        if config.get('enable_double_chance', False):
            dc_min = config.get('dc_min_prob', 0.75)
            dc_sec = config.get('dc_secondary_threshold', 0.80)
            analysis['market_focus'].append(f'Double Chance (min: {dc_min}, secondary: {dc_sec})')

            if dc_min >= 0.8:
                analysis['recommendations'].append('Very conservative DC thresholds - consider lowering for more opportunities')
            elif dc_min <= 0.65:
                analysis['recommendations'].append('Aggressive DC thresholds - ensure quality control')
        else:
            analysis['recommendations'].append('Consider enabling Double Chance for parlay safety')

        return analysis

    def generate_optimal_combinations(self) -> Dict[str, List[str]]:
        """Generate optimal parameter combinations for different scenarios"""
        return {
            'weekend_accumulator': [
                '--min-confidence 0.7',
                '--ml-mode predict',
                '--enable-double-chance',
                '--dc-min-prob 0.78',
                '--dc-secondary-threshold 0.85',
                '--dc-allow-multiple',
                '--verbose'
            ],

            'daily_value_hunter': [
                '--min-confidence 0.62',
                '--ml-mode predict',
                '--enable-double-chance',
                '--dc-min-prob 0.72',
                '--dc-secondary-threshold 0.78',
                '--dc-allow-multiple',
                '--verbose'
            ],

            'safe_builder': [
                '--min-confidence 0.75',
                '--ml-mode predict',
                '--enable-double-chance',
                '--dc-min-prob 0.82',
                '--dc-secondary-threshold 0.88',
                '--dc-allow-multiple false',
                '--verbose'
            ],

            'ml_edge_exploiter': [
                '--min-confidence 0.58',
                '--ml-mode predict',
                '--enable-double-chance',
                '--dc-min-prob 0.68',
                '--dc-secondary-threshold 0.75',
                '--dc-allow-multiple',
                '--verbose'
            ]
        }

    def print_parameter_guide(self):
        """Print comprehensive parameter guide"""
        print("=" * 80)
        print("FOOTBALL ANALYTICS PARAMETER GUIDE")
        print("=" * 80)
        print()

        for param_name, config in self.parameters.items():
            print(f"Parameter: {config.name}")
            print(f"Description: {config.description}")
            print(f"Type: {config.data_type}")
            print(f"Default: {config.default_value}")
            print(f"Range: {config.valid_range}")
            print(f"Markets Affected: {', '.join([m.value for m in config.market_impact])}")
            print(f"Strategic Purpose: {config.strategic_purpose}")
            print(f"ML Enhancement: {config.ml_enhancement}")
            print(f"Parlay Considerations: {config.parlay_considerations}")
            print("-" * 80)
            print()

    def print_market_combinations(self):
        """Print optimal market combination strategies"""
        print("=" * 80)
        print("OPTIMAL PARAMETER COMBINATIONS")
        print("=" * 80)
        print()

        for strategy_name, strategy in self.market_combinations.items():
            print(f"Strategy: {strategy_name.upper()}")
            print(f"Description: {strategy['description']}")
            print(f"Target Markets: {', '.join(strategy['target_markets'])}")
            print(f"Risk Level: {strategy['risk_level']}")
            print(f"Expected Odds: {strategy['expected_odds_range']}")
            print(f"Parlay Size: {strategy['typical_parlay_size']}")
            print("Parameters:")
            for param, value in strategy['parameters'].items():
                print(f"  {param} {value}")
            print("-" * 80)
            print()

    def print_ml_advantages(self):
        """Print ML enhancement details"""
        print("=" * 80)
        print("MACHINE LEARNING ADVANTAGES")
        print("=" * 80)
        print()

        for market_type, advantage in self.ml_advantages.items():
            print(f"Market: {market_type.replace('_', ' ').title()}")
            print(advantage.strip())
            print("-" * 80)
            print()

def main():
    """Main function to run the parameter analysis guide"""
    guide = FootballAnalyticsParameterGuide()

    print("Football Analytics Parameter Analysis Guide")
    print("Generated on:", "December 7, 2025")
    print()

    # Print sections
    guide.print_parameter_guide()
    guide.print_market_combinations()
    guide.print_ml_advantages()

    # Analyze the user's current configuration
    print("=" * 80)
    print("ANALYSIS OF YOUR CURRENT CONFIGURATION")
    print("=" * 80)

    current_config = {
        'min_confidence': 0.6,
        'ml_mode': 'predict',
        'enable_double_chance': True,
        'dc_min_prob': 0.75,
        'dc_secondary_threshold': 0.80,
        'dc_allow_multiple': True,
        'verbose': True
    }

    analysis = guide.analyze_current_configuration(current_config)

    print("Configuration Analysis:")
    for key, value in analysis.items():
        if key != 'configuration':
            print(f"{key.replace('_', ' ').title()}: {value}")
    print()

    # Print optimal combinations
    print("=" * 80)
    print("RECOMMENDED PARAMETER COMBINATIONS")
    print("=" * 80)

    combinations = guide.generate_optimal_combinations()
    for scenario, params in combinations.items():
        print(f"\n{scenario.upper()}:")
        print("Command: python cli.py --task full-league --league E0 " + " ".join(params))
        print()

if __name__ == "__main__":
    main()
