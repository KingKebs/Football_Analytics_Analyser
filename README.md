# Football Analytics Analyser

A sophisticated football betting analysis system using statistical modeling and machine learning techniques.

## Features

- **Multi-League Support**: Analyze 25+ European leagues including EPL, Bundesliga, La Liga, Serie A, etc.
- **Full League Analysis**: Analyze entire league rounds, generate parlays for upcoming games
- **Advanced Algorithms**: 11 core algorithms including Poisson distribution, Kelly criterion, and xG modeling
- **Recent Form Analysis**: Exponential decay weighting of recent matches
- **Smart Caching**: Rate-limited data fetching to respect API limits
- **Conservative Betting**: Designed for small-stakes punters with risk management

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python automate_football_analytics.py
```

The system will prompt you to select a league and teams for analysis.

### Full League Analysis

```bash
python automate_football_analytics_fullLeague.py --league E0 --bankroll 100
```

This script analyzes all upcoming matches in the selected league, generates suggestions for each match, and selects favorable parlays with low risk and high returns.

### Web Output Examples

```bash
python web_output_examples.py
```

This script demonstrates various ways to format the full league analysis results for web applications, including JSON API responses, HTML dashboards, and CSV exports.

## Supported Leagues

- **English**: Premier League (E0), Championship (E1), League One (E2), League Two (E3)
- **German**: Bundesliga (D1), 2. Bundesliga (D2)
- **Spanish**: La Liga (SP1), Segunda División (SP2)
- **Italian**: Serie A (I1), Serie B (I2)
- **French**: Ligue 1 (F1), Ligue 2 (F2)
- **Portuguese**: Primeira Liga (P1)
- **Dutch**: Eredivisie (N1)
- **Scottish**: Premiership (SC0), Championship (SC1), League One (SC2), League Two (SC3)
- **Belgian**: Pro League (B1)
- **Greek**: Super League (G1)
- **Turkish**: Süper Lig (T1)

## Project Structure

```
Football_Analytics_Analyser/
├── README.md
├── README_Downloader_Script.md
├── ALGORITHMS.md
├── requirements.txt
├── organize_structure.sh
├── organize_structure.py
├── web_output_examples.py
├── algorithms.py
├── automate_football_analytics.py
├── automate_football_analytics_fullLeague.py
├── download_all_tabs.py
├── visualize_score_table.py
├── data/
│   ├── home_away_team_strengths_*.csv
│   ├── league_data_*.csv
│   ├── suggestion_*.json
│   └── old csv/
│       └── *.csv
├── football-data/
│   ├── data.zip
│   └── all-euro-football/
│       └── *.csv
├── tmp/
│   └── *.log
├── __pycache__/
│   └── *.pyc
├── visuals/
│   ├── *.png
│   └── *.csv
├── venv/
├── .git/
├── .gitignore
└── .idea/
```

### File Descriptions

- `automate_football_analytics.py` - Main analysis script for individual match predictions
- `automate_football_analytics_fullLeague.py` - Full league analysis script for upcoming games and parlay generation
- `algorithms.py` - Core mathematical algorithms for predictions
- `download_all_tabs.py` - Data fetching utilities
- `visualize_score_table.py` - Visualization tools for score tables
- `organize_structure.sh` - Bash script to create directories and organize files according to project structure
- `organize_structure.py` - Python script to create directories and organize files according to project structure
- `web_output_examples.py` - Script for generating web output examples
- `ALGORITHMS.md` - Detailed documentation of all algorithms
- `README.md` - Main project documentation
- `README_Downloader_Script.md` - Documentation for the downloader script
- `requirements.txt` - Python dependencies

## Algorithm Overview

1. **Team Strength Calculation** - Normalizes performance vs league average
2. **Expected Goals (xG) Estimation** - Multiplicative model with home advantage
3. **Poisson Distribution** - Mathematical foundation for goal modeling
4. **Score Probability Matrix** - All possible match score probabilities
5. **Market Probability Extraction** - Converts scores to betting markets
6. **Recent Form Analysis** - Exponential decay weighting
7. **Form-Season Blending** - Weighted average of stats and form
8. **Kelly Criterion Staking** - Optimal bet sizing with caps
9. **Parlay Generation** - Value-ranked combination bets
10. **Smart Caching** - Rate limiting with graceful degradation
11. **Heuristic Estimation** - Secondary market predictions

## New CLI options (risk and confidence)

Both main CLIs now support risk and confidence controls and log suggested bet slips:

- `--min-confidence FLOAT` (default 0.6): minimum probability to surface a market from the Poisson score matrix.
- `--risk-profile {conservative|moderate|aggressive}` (default moderate): controls Kelly scaling caps for stake suggestions.
  - conservative: risk_multiplier=0.25, f_max=0.01
  - moderate:    risk_multiplier=0.5,  f_max=0.015
  - aggressive:  risk_multiplier=0.9,  f_max=0.03

Suggested parlays are logged to `data/bet_history.csv` with timestamp, legs, probability, odds, stake_suggestion, and parameters used.

### Examples

```bash
# Single match interactive flow (choose league and teams interactively)
python3 automate_football_analytics.py --bankroll 500 --risk-profile conservative --min-confidence 0.65

# Full league flow for EPL, using parsed fixtures when available
python3 automate_football_analytics_fullLeague.py --league E0 --bankroll 500 --risk-profile moderate --min-confidence 0.65

# Analyze all parsed fixtures across leagues
python3 automate_football_analytics_fullLeague.py --use-parsed-all --bankroll 500 --risk-profile aggressive --min-confidence 0.7
```

> Note: Bet history logging appends to `data/bet_history.csv`. You can analyze it with pandas or a spreadsheet to review stake sizing and outcomes.

## Rating model (goal-supremacy) and blending

You can enable a rating-driven 1X2 model based on recent goal-supremacy and optionally blend it with Poisson-derived 1X2 probabilities.

CLI flags (both single-match and full-league scripts):
- `--rating-model {none,goal_supremacy,blended}` (default `none`)
- `--rating-last-n INT` (default `6`)
- `--min-sample-for-rating INT` (default `30`)
- `--rating-blend-weight FLOAT` (default `0.3`)

Examples:
```bash
# Use pure rating-based 1X2 on single match flow
python3 automate_football_analytics.py --league E0 --rating-model goal_supremacy --rating-last-n 6

# Blend rating (30%) with Poisson (70%) on full league flow
python3 automate_football_analytics_fullLeague.py --league E0 --rating-model blended --rating-blend-weight 0.3 --rating-last-n 6
```

### Backtesting and diagnostics

Use the analyzer to compare suggestions to known results and to evaluate strike-rate/ROI by rating bins:

```bash
# Accuracy and parlay diagnostics only
python3 analyze_suggestions_results.py --suggestions data/full_league_suggestions_E0_20251102_095227.json --results data/sample_results_20251102.csv

# Include rating-bin backtest (uses historical CSVs in data/old\ csv/)
python3 analyze_suggestions_results.py --suggestions data/full_league_suggestions_E0_20251102_095227.json \
  --results data/sample_results_20251102.csv \
  --backtest-rating --rating-last-n 6 --rating-bins "-10,-6,-4,-2,-1,0,1,2,4,6,10"
```

Notes:
- Historical files should be placed under `data/old csv/` and include columns Date, HomeTeam, AwayTeam, FTHG, FTAG (common football-data columns are auto-normalized).
- When `--rating-model=none`, the behavior is unchanged from prior Poisson-only logic.

## Disclaimer

This system is designed for educational purposes and conservative betting strategies. It does not guarantee profits and should be used responsibly.
