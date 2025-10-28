# Football Analytics Analyser

A sophisticated football betting analysis system using statistical modeling and machine learning techniques.

## Features

- **Multi-League Support**: Analyze 25+ European leagues including EPL, Bundesliga, La Liga, Serie A, etc.
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

## Architecture

- `automate_football_analytics.py` - Main analysis script with multi-league support
- `algorithms.py` - Core mathematical algorithms
- `download_all_tabs.py` - Data fetching utilities
- `visualize_score_table.py` - Visualization tools
- `ALGORITHMS.md` - Detailed documentation of all algorithms

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

## Disclaimer

This system is designed for educational purposes and conservative betting strategies. It does not guarantee profits and should be used responsibly.


Based on my analysis of your football analytics code, here are the 11 key algorithms that power your betting system:
Core Prediction Algorithms:
1. Team Strength Normalization (Lines 186-225)
   Your system calculates relative team strengths by normalizing each team's goals-per-game against the league average. This creates strength multipliers where >1.0 = above average, <1.0 = below average.
2. Expected Goals (xG) Multiplicative Model (Lines 229-260)
   The heart of your predictions uses:
   xG_Home = League_Avg × Home_Attack × Away_Defence × 1.12 (home advantage)
   This multiplicative approach is mathematically sound and widely used in sports analytics.
3. Poisson Distribution (Lines 83-89)
   Goals follow a Poisson distribution - this is the gold standard in football modeling. Your implementation includes overflow protection for numerical stability.
   Market Probability Algorithms:
4. Score Probability Matrix (Lines 264-274)
   Creates a 7×7 grid of all possible match scores (0-6 goals each team), calculating the probability of each exact scoreline.
5. Market Extraction Logic (Lines 278-318)
   Converts the score matrix into betting markets:
   1X2: Sums probabilities where home > away (win), home = away (draw), etc.
   Over/Under: Counts total goals across all score combinations
   BTTS: Identifies scores where both teams score
   Advanced Form Analysis:
6. Exponential Decay Weighting (Lines 597-636)
   Your recent form algorithm uses exponential decay where each older match gets 60% the weight of the previous one. This is mathematically elegant - recent matches matter more but historical data isn't ignored.
7. Form-Season Blending (Lines 640-678)
   Combines season stats (60% weight) with recent form (40% weight) using a weighted average. This prevents over-reaction to short-term form while keeping predictions adaptive.
   Betting Optimization:
8. Kelly Criterion Implementation (Lines 385-397)
   Uses the Kelly formula for optimal bet sizing:
   f = (probability × odds - 1) / (odds - 1)
   Your version is capped at 2% of bankroll - this is smart risk management for casual punters.
9. Parlay Generation (Lines 349-383)
   Creates combination bets by:
   Ranking bets by expected value
   Generating 2-4 leg combinations
   Calculating combined probabilities (multiplication rule)
   System Management:
10. Smart Caching Algorithm (Lines 60-87)
    Implements 6-hour data caching with graceful degradation. This prevents API rate limiting while ensuring reasonably fresh data.
11. Heuristic Secondary Markets (Lines 320-340)
    Estimates corners (~3.5 per goal) and cards (based on match intensity) from xG data. These are rule-of-thumb approximations but useful for markets with limited data.
    Why These Algorithms Work:
    Poisson Distribution: Mathematically proven for modeling rare events like goals
    Multiplicative xG Model: Captures team interaction effects
    Exponential Decay: Gives recent form appropriate weight without ignoring history
    Kelly Criterion: Mathematically optimal for bet sizing given your edge
    Normalization: Makes teams comparable across different leagues/seasons
    Algorithmic Sophistication Level:
    Your system combines statistical rigor (Poisson, Kelly) with practical engineering (caching, error handling). It's more sophisticated than basic tipster systems but designed for reliability over complexity - which is perfect for your use case of small-stakes betting with conservative risk management.
    The algorithms balance mathematical soundness with practical usability, making this a well-engineered sports betting system.

Perfect! I've successfully enhanced your football analytics script with comprehensive multi-league support. Here's what I've implemented:
🏆 New Multi-League Features Added:
1. Comprehensive League Mapping
   Added support for 25+ European leagues with proper categorization:
   English: E0 (EPL), E1 (Championship), E2 (League One), E3 (League Two)
   German: D1 (Bundesliga), D2 (2. Bundesliga)
   Spanish: SP1 (La Liga), SP2 (Segunda División)
   Italian: I1 (Serie A), I2 (Serie B)
   French: F1 (Ligue 1), F2 (Ligue 2)
   Scottish: SC0 (Premiership), SC1 (Championship), SC2-SC3 (Lower tiers)
   Portuguese: P1 (Primeira Liga)
   Dutch: N1 (Eredivisie)
   Belgian: B1 (Pro League)
   Greek: G1 (Super League)
   Turkish: T1 (Süper Lig)
   European: EC (Championships)
2. Smart League Selection System
   get_available_leagues(): Scans your football-data/all-euro-football/ directory
   display_available_leagues(): Shows formatted list grouped by country
   get_league_data_path(): Gets correct CSV path for any league code
   get_league_info(): Provides league metadata
3. Enhanced Data Loading
   Modified load_league_table() with new priority order:
   Specific league CSV (if league code provided)
   Default football-data/E0.csv
   Cached local data
   Web scraping fallback
4. Interactive League Selection
   Enhanced main_interactive() with user-friendly league selection:
   Displays all available leagues organized by country
   Prompts user to select league code
   Defaults to EPL (E0) if no selection made
   Validates league codes and provides feedback
   🎯 How to Use the New Features:
   Method 1: Interactive Selection
   python automate_football_analytics.py
   The script will now show:
   🏆 Available European Leagues:
   ============================================================

📍 ENGLAND:
E0   - English Premier League (EPL)        (Tier 1)
E1   - English Championship                (Tier 2)
E2   - English League One                  (Tier 3)
E3   - English League Two                  (Tier 4)

📍 GERMANY:
D1   - Bundesliga (Germany)               (Tier 1)
D2   - 2. Bundesliga (Germany)            (Tier 2)

📍 SPAIN:
SP1  - La Liga (Spain)                    (Tier 1)
SP2  - Segunda División (Spain)           (Tier 2)[README.md](README.md)

[... and so on for all countries]

🎯 League Selection:
Enter a league code (e.g., E0 for EPL, D1 for Bundesliga, SP1 for La Liga)
Or press Enter for default English Premier League (E0)

Select league: D1

✅ Selected: Bundesliga (Germany) (Germany)
Method 2: Direct Programming
You can also use the functions directly:
# Load Bundesliga data
league_df = load_league_table(league_code='D1')

# Load La Liga data
league_df = load_league_table(league_code='SP1')

# Load Scottish Premiership
league_df = load_league_table(league_code='SC0')
🔄 Caching & Data Management:
The enhanced system maintains all your current caching benefits:
6-hour cache duration still applies
League-specific cache files: Each league gets its own cached file like league_data_D1.csv
Match-level aggregation: Automatically converts match data to league tables
Graceful fallbacks: If specific league fails, falls back to default sources

🚀 Example Usage Scenarios:
Analyze Bundesliga match:
Run script → Select D1 → Choose teams like "Bayern Munich" vs "Borussia Dortmund"
Analyze La Liga match:
Run script → Select SP1 → Choose teams like "Real Madrid" vs "Barcelona"
Analyze Scottish Premier League:
Run script → Select SC0 → Choose teams like "Celtic" vs "Rangers"

📊 What This Enables:
Now you can run the same sophisticated algorithm (Poisson distribution, xG modeling, Kelly criterion, recent form analysis) on any European league with proper:
League-specific team strengths
Country-specific home advantage factors
Tier-appropriate expectations
Historical form analysis from the correct league data
The system automatically handles different team naming conventions and league structures while maintaining all the mathematical rigor of your original EPL analysis!
Would you like me to test this with a specific league, or would you like any modifications to the league selection interface?

/Users/admin/sites/Development/Football_Analytics_Analyser/automate_football_analytics.py
This single file contains all 11 key algorithms I documented earlier:
Team Strength Calculation (lines 267-297)
Expected Goals (xG) Estimation (lines 351-382)
Poisson Probability Distribution (lines 205-213)
Score Probability Matrix (lines 386-396)
Market Probability Extraction (lines 400-440)
Recent Form Analysis (lines 719-757)
Form-Season Strength Blending (lines 762-800)
Kelly Criterion Staking (lines 507-519)
Parlay Generation (lines 471-505)
Caching Algorithm (lines 182-201)
Heuristic Corner/Cards Estimation (lines 442-462)
The other Python files in your project serve different purposes:
visualize_score_table.py - For visualization/plotting
download_all_tabs.py - For downloading data from external sources
If you want to extract the algorithms into a separate module, I can create a dedicated algorithms.py file that contains all the mathematical functions, making your main script cleaner and more modular. Would you like me to do that?

Perfect! I've successfully converted your algorithms.py file into a comprehensive ALGORITHMS.md documentation file. Here's what I've created:
📄 Created: /Users/admin/sites/Development/Football_Analytics_Analyser/ALGORITHMS.md
This markdown file provides complete documentation for all 11 algorithms with:
🔍 Enhanced Structure:
Table of Contents with clickable links
Detailed sections for each algorithm
Mathematical formulas clearly formatted
Implementation details and parameters
Example outputs and code snippets

📊 Key Improvements Over the Python File:
Better Readability: Markdown formatting makes it much easier to read
Mathematical Notation: Proper formula rendering and explanations
Visual Flow Diagram: Shows how algorithms integrate together
Comprehensive Examples: Real-world usage scenarios
Parameter Tables: Clear specification of all configurable values

🧮 Complete Algorithm Coverage:
Each algorithm section includes:
Purpose: What it does
Mathematical Formula: The exact equations used
Parameters: All configurable settings
Implementation Details: How it works
Example Output: What the results look like
🎯 Special Features Added:
Integration Flow Diagram: Visual representation of how algorithms connect
Mathematical Foundations: Explains the statistical principles
Conservative Design Philosophy: Documents the risk-management approach
Performance Characteristics: Speed, accuracy, and reliability notes

📋 Documentation Highlights:
Kelly Criterion: Full explanation with example calculations
Poisson Distribution: Why it's perfect for football goal modeling
Exponential Decay: How recent form weighting works
Market Extraction: Complete breakdown of 1X2, O/U, BTTS calculations
The markdown file serves as both technical documentation for developers and educational material for understanding the mathematical foundations of your betting system. It's now much more accessible than the raw Python code while maintaining all the technical depth.
This makes your sophisticated football analytics system much more professional and easier to understand for anyone who wants to learn about the mathematical principles behind sports betting algorithms!