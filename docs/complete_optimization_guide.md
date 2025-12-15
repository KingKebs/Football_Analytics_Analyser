====================================================================================================
FOOTBALL ANALYTICS PARAMETER OPTIMIZATION SUITE
====================================================================================================
Based on analysis of your actual data patterns

CURRENT SYSTEM PERFORMANCE ANALYSIS:
--------------------------------------------------
Double Chance Selection Rate: 26.1%
Average DC Probability: 0.786
ML Home Team Edge: +0.085
ML Away Team Edge: -0.087
Most Selected DC Type: 12

OPTIMIZED PARAMETER STRATEGIES:
--------------------------------------------------

Strategy: Conservative Multi-Parlay
Description: High-probability selections for 5-8 leg parlays focusing on DC and safe markets
Risk Level: Low (70-80% individual leg success rate)
Target ROI: 15-25% per successful parlay
Parameters:
  --min-confidence 0.72
  --ml-mode predict
  --enable-double-chance True
  --dc-min-prob 0.82
  --dc-secondary-threshold 0.87
  --dc-allow-multiple False
  --verbose True
Expected Outcomes:
  selections_per_day: 8-12 across all leagues
  avg_odds_per_leg: 1.18-1.35
  parlay_odds_5_legs: 2.4-4.5
  parlay_odds_8_legs: 4.3-10.8
  hit_rate_estimate: 12-18% (parlays)
Market Focus: DC (primarily 12), BTTS No, Under 3.5, Over 0.5
--------------------------------------------------------------------------------

Strategy: Balanced Value Hunter
Description: ML-enhanced value detection with mixed markets for 3-5 leg parlays
Risk Level: Medium (60-75% individual leg success rate)
Target ROI: 25-40% per successful parlay
Parameters:
  --min-confidence 0.68
  --ml-mode predict
  --enable-double-chance True
  --dc-min-prob 0.78
  --dc-secondary-threshold 0.83
  --dc-allow-multiple True
  --verbose True
Expected Outcomes:
  selections_per_day: 12-18 across all leagues
  avg_odds_per_leg: 1.25-1.65
  parlay_odds_3_legs: 2.0-4.5
  parlay_odds_5_legs: 3.1-11.4
  hit_rate_estimate: 15-25% (parlays)
Market Focus: DC (mixed), BTTS Yes, Over 2.5, 1X2 (ML edge), Over 1.5
--------------------------------------------------------------------------------

Strategy: ML Edge Exploiter
Description: Maximum ML advantage exploitation for 2-4 leg parlays with higher variance
Risk Level: Medium-High (55-70% individual leg success rate)
Target ROI: 40-80% per successful parlay
Parameters:
  --min-confidence 0.62
  --ml-mode predict
  --enable-double-chance True
  --dc-min-prob 0.72
  --dc-secondary-threshold 0.78
  --dc-allow-multiple True
  --verbose True
Expected Outcomes:
  selections_per_day: 15-25 across all leagues
  avg_odds_per_leg: 1.35-2.2
  parlay_odds_2_legs: 1.8-4.8
  parlay_odds_4_legs: 3.3-23.4
  hit_rate_estimate: 18-30% (parlays)
Market Focus: 1X2 (ML favored), BTTS (ML edge), Over/Under (ML adjusted), DC (value spots)
--------------------------------------------------------------------------------

Strategy: Daily Accumulator Builder
Description: Daily 6-10 leg accumulators focusing on same-day fixtures with DC safety
Risk Level: High (individual legs), but safer market selection
Target ROI: 50-200% per successful accumulator
Parameters:
  --min-confidence 0.75
  --ml-mode predict
  --enable-double-chance True
  --dc-min-prob 0.85
  --dc-secondary-threshold 0.9
  --dc-allow-multiple False
  --verbose True
Expected Outcomes:
  selections_per_day: 6-10 from day's fixtures
  avg_odds_per_leg: 1.15-1.28
  accumulator_odds_6_legs: 2.5-5.2
  accumulator_odds_10_legs: 4.0-14.6
  hit_rate_estimate: 8-15% (accumulators)
Market Focus: DC (ultra-safe), BTTS No (strong defenses), Under 2.5 (tight games)
--------------------------------------------------------------------------------

MARKET-SPECIFIC OPTIMIZATIONS:
--------------------------------------------------

Configuration: double_chance_specialist
Description: Maximizes Double Chance opportunities based on your 26.1% selection rate
Rationale: Your data shows DC avg prob of 0.786, suggesting current thresholds are working
Optimized Parameters:
  --min-confidence 0.6
  --ml-mode predict
  --enable-double-chance True
  --dc-min-prob 0.75
  --dc-secondary-threshold 0.82
  --dc-allow-multiple True
  --verbose True
Expected Improvement: +15-20% more DC selections while maintaining quality


Configuration: btts_value_hunter
Description: Optimizes BTTS selection (your avg: 0.692 prob, 1.45 odds)
Rationale: Strong BTTS performance suggests room for more aggressive selection
Optimized Parameters:
  --min-confidence 0.65
  --ml-mode predict
  --enable-double-chance False
  --verbose True
Expected Improvement: Better BTTS identification through ML edge detection


Configuration: ml_home_advantage
Description: Exploits ML +0.085 home edge vs Poisson models
Rationale: Your ML shows significant home team bias - 14 large home favors detected
Optimized Parameters:
  --min-confidence 0.58
  --ml-mode predict
  --enable-double-chance True
  --dc-min-prob 0.7
  --dc-secondary-threshold 0.75
  --dc-allow-multiple True
  --verbose True
Expected Improvement: Capture more home team value where ML beats traditional models

PARLAY MATHEMATICS & EDGE ANALYSIS:
--------------------------------------------------
Probability Mathematics:
  conservative_parlay_5_legs:
    Individual Leg Probability: 0.78
    Parlay Success Rate: 28.9%
    Break-even Odds: 3.46
  balanced_parlay_4_legs:
    Individual Leg Probability: 0.7
    Parlay Success Rate: 24.0%
    Break-even Odds: 4.16
  aggressive_parlay_3_legs:
    Individual Leg Probability: 0.65
    Parlay Success Rate: 27.5%
    Break-even Odds: 3.64

ML Edge Analysis:
  Home Edge Value: +0.085
  Away Edge Value: -0.087
  Potential Roi Boost: 5-15% when ML edges are properly identified and exploited
  Edge Frequency: Significant edges in ~64% of ML comparisons

Bankroll Management:
  Conservative Strategy: 2-3% per parlay, higher volume
  Balanced Strategy: 3-5% per parlay, medium volume
  Aggressive Strategy: 5-8% per parlay, lower volume
  Ml Edge Strategy: 4-6% when edge detected, otherwise 2-3%

====================================================================================================
READY-TO-USE CLI COMMANDS:
====================================================================================================

CONSERVATIVE MULTI-PARLAY (Recommended for beginners):
E0: python cli.py --task full-league --league E0 --min-confidence 0.72 --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose
E1: python cli.py --task full-league --league E1 --min-confidence 0.72 --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose
D1: python cli.py --task full-league --league D1 --min-confidence 0.72 --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose
F1: python cli.py --task full-league --league F1 --min-confidence 0.72 --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose
I1: python cli.py --task full-league --league I1 --min-confidence 0.72 --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose
SP1: python cli.py --task full-league --league SP1 --min-confidence 0.72 --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose
B1: python cli.py --task full-league --league B1 --min-confidence 0.72 --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose

BALANCED VALUE HUNTER (Recommended for experienced):
E0: python cli.py --task full-league --league E0 --min-confidence 0.68 --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose
E1: python cli.py --task full-league --league E1 --min-confidence 0.68 --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose
D1: python cli.py --task full-league --league D1 --min-confidence 0.68 --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose
F1: python cli.py --task full-league --league F1 --min-confidence 0.68 --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose
I1: python cli.py --task full-league --league I1 --min-confidence 0.68 --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose
SP1: python cli.py --task full-league --league SP1 --min-confidence 0.68 --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose
B1: python cli.py --task full-league --league B1 --min-confidence 0.68 --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose

ML EDGE EXPLOITER (Advanced users):
E0: python cli.py --task full-league --league E0 --min-confidence 0.62 --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose
E1: python cli.py --task full-league --league E1 --min-confidence 0.62 --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose
D1: python cli.py --task full-league --league D1 --min-confidence 0.62 --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose
F1: python cli.py --task full-league --league F1 --min-confidence 0.62 --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose
I1: python cli.py --task full-league --league I1 --min-confidence 0.62 --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose
SP1: python cli.py --task full-league --league SP1 --min-confidence 0.62 --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose
B1: python cli.py --task full-league --league B1 --min-confidence 0.62 --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose
