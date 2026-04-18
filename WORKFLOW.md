# Workflow Documentation - Complete Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Full League Analysis](#full-league-analysis)
3. [Workflow Integration](#workflow-integration)
4. [Streamlit Dashboard](#streamlit-dashboard)

---

## Quick Start

### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate

# Verify dependencies
pip list | grep -E "streamlit|pandas|numpy|xgboost"
```

### Basic Workflow
```bash
# 1. Run full league analysis
python3 cli.py full-league --league E0 --ml-mode off

# 2. Launch Streamlit dashboard
streamlit run src/streamlit_app.py

# 3. Open browser to http://localhost:8501
# 4. Navigate to "Full League Suggestions"
# 5. Check "Combo Market Opportunities" section
```

---

## Full League Analysis

### Command Syntax
```bash
python3 cli.py full-league [OPTIONS]
```

### Common Options

#### League Selection
```bash
# Single league
python3 cli.py full-league --league E0

# Multiple leagues (in one run)
# Note: Specify once, runs all configured leagues if no --league specified

# All available leagues
python3 cli.py full-league
```

Available league codes:
- **E0**: English Premier League
- **E1**: English Championship
- **E2**: English League Two
- **E3**: English League One
- **D1**: German Bundesliga
- **F1**: French Ligue 1
- **I1**: Italian Serie A
- **SP1**: Spanish La Liga
- **N1**: Dutch Eredivisie

#### ML Mode
```bash
# Poisson analysis only (no ML)
python3 cli.py full-league --league E0 --ml-mode off

# Use pre-trained models for predictions
python3 cli.py full-league --league E0 --ml-mode predict

# Train models from historical data, then predict
python3 cli.py full-league --league E0 --ml-mode train
```

#### Confidence Threshold
```bash
# Higher threshold = fewer picks (only strongest predictions)
python3 cli.py full-league --league E0 --min-confidence 0.75

# Lower threshold = more picks (include moderate predictions)
python3 cli.py full-league --league E0 --min-confidence 0.50

# Default: 0.60 (60% confidence)
```

#### Double Chance Markets
```bash
# Enable double chance (1X, X2, 12) market predictions
python3 cli.py full-league --league E0 --enable-double-chance

# Configure DC thresholds
python3 cli.py full-league --league E0 --enable-double-chance \
  --dc-min-prob 0.75 --dc-secondary-threshold 0.80
```

### Output Files

After running analysis:

```
data/analysis/
├── full_league_suggestions_E0_20260418_094111.json
│   └── Per-league detailed results (all matches, markets, picks)
├── full_league_suggestions_E0_20260418_094111_formatted.txt
│   └── Human-readable formatted output
├── consolidated_full_league_20260418_20260418_094111.json
│   └── All leagues combined in one file
├── consolidated_full_league_20260418_20260418_094111_formatted.txt
│   └── Formatted consolidated view
└── cross_league_summary_20260418_094111.json
    └── Cross-league parlay summary
```

---

## Workflow Integration

### Data Processing Pipeline

```
1. INPUT: Team Fixtures
   ├─ Teams and scheduled matches
   ├─ Match dates
   └─ Competition/league

2. FETCH: Historical Data
   ├─ Download league tables
   ├─ Get historical match results
   └─ Extract team performance metrics

3. ANALYZE: Match-by-Match
   ├─ Estimate expected goals (xG)
   ├─ Build score probability matrix
   ├─ Extract markets (1X2, OU, BTTS, DC)
   ├─ Extract combo markets
   └─ Detect combo value opportunities

4. PROCESS: ML Integration (if enabled)
   ├─ Train models OR load pre-trained models
   ├─ Make predictions for each match
   ├─ Compare ML vs Poisson
   └─ Generate prediction signatures

5. OUTPUT: Results
   ├─ Save per-league JSON
   ├─ Create consolidated file
   ├─ Generate text reports
   └─ Generate cross-league parlays

6. DISPLAY: Streamlit Dashboard
   ├─ Load and filter results
   ├─ Display full league suggestions
   ├─ Show ML predictions
   ├─ Show combo opportunities
   └─ Display corner analysis (if available)
```

### Process Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      FULL LEAGUE ANALYSIS                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
   [FIXTURES]                         [LEAGUE TABLES]
   [LOAD MATCHES]                     [TEAM STRENGTHS]
        ↓                                   ↓
        └─────────────────┬─────────────────┘
                          ↓
              ┌───────────────────────┐
              │  PROCESS EACH MATCH   │
              └───────────────────────┘
                          ↓
         ┌────────────────┼────────────────┐
         ↓                ↓                ↓
    [XG EST]      [SCORE MATRIX]    [MARKETS]
         ↓                ↓                ↓
         └────────────────┼────────────────┘
                          ↓
         ┌────────────────┼────────────────┐
         ↓                ↓                ↓
    [1X2 PICKS]    [COMBOS]         [PARLAYS]
         ↓                ↓                ↓
         └────────────────┼────────────────┘
                          ↓
              ┌───────────────────────┐
              │   CONSOLIDATE DATA    │
              │  (ALL LEAGUES)        │
              └───────────────────────┘
                          ↓
         ┌────────────────┼────────────────┐
         ↓                ↓                ↓
    [SAVE JSON]    [SAVE TEXT]     [CROSS-LEAGUE]
         ↓                ↓                ↓
         └────────────────┼────────────────┘
                          ↓
            ┌─────────────────────────┐
            │  STREAMLIT DISPLAY      │
            │  - Full League          │
            │  - Combos               │
            │  - ML Predictions       │
            │  - Corner Analysis      │
            └─────────────────────────┘
```

---

## Streamlit Dashboard

### Navigation

#### Tab 1: Full League Suggestions
- **Purpose**: View all match suggestions from selected league or consolidated file
- **Displays**: 
  - Match details (home/away teams)
  - Estimated xG
  - Top suggested picks
  - **Combo Market Opportunities** (if available)
  - Download buttons

- **Interaction**:
  1. Select file from dropdown
  2. View matches in table
  3. Check combo opportunities
  4. Download as JSON

#### Tab 2: ML Predictions
- **Purpose**: Compare ML model predictions with Poisson baseline
- **Displays**:
  - ML total goals predictions
  - 1X2 probabilities from ML
  - BTTS probabilities from ML
  - Double Chance derived odds
  - ML vs Poisson deltas
  - Prediction signatures

- **Filters**:
  - Team name search
  - League selection
  - Show ML vs Poisson comparison toggle

#### Tab 3: Corner Analysis
- **Purpose**: Analyze corner kick predictions and statistics
- **Displays**:
  - Corner distribution
  - Feature correlations
  - Team statistics
  - First half/second half ratios
  - Market line recommendations

- **Selection**:
  - League selection
  - Team search
  - Minimum corners filter

#### Tab 4: Corner Predictions
- **Purpose**: View predicted corner kicks for upcoming matches
- **Displays**:
  - Expected corners (home/away)
  - Total corners prediction
  - First half ratios
  - Market recommendations (Over/Under)
  - Half-split analysis

- **Features**:
  - League multiselect
  - Minimum corners threshold
  - Team search

#### Tab 5: Live Corner Predictor
- **Purpose**: Real-time corner predictions for custom match-ups
- **Features** (under development):
  - League selection
  - Home/away team selection
  - Live prediction generation

---

### Combo Market Opportunities Section

#### Display Elements

```
🎯 Combo Market Opportunities

┌─────────────────────────────────────────────────────────────┐
│ League │ Match │ Combo Type │ Selection │ Prob │ Odds │ EV%│
├─────────────────────────────────────────────────────────────┤
│ E0     │ Team A vs Team B │ 1X2_OU2.5 │ Home_Under │ 22% │ 2.86│ +1.2%│
│ E0     │ Team C vs Team D │ 1X2_BTTS │ Away_Yes │ 45% │ 2.22│ +0.1%│
└─────────────────────────────────────────────────────────────┘

Summary Metrics:
🎲 Total Combos: 2
📊 Avg EV %: +0.65%
⭐ High Value: 1
🎯 Avg Odds: 2.54
```

#### Interpretation

- **Combo Type**: Type of market combination
  - `1X2_OU`: Match result + goal total
  - `1X2_BTTS`: Match result + both teams scoring
  - `OU_BTTS`: Goal total + both teams scoring
  - `DC_BTTS`: Double chance + both teams scoring

- **Selection**: Specific combo option chosen
  - Example: `Home_Under2.5` = Home wins AND total ≤ 2.5

- **Probability**: Our calculated probability (not odds)
  - Based on score matrix correlation

- **Odds**: Decimal odds (what you'd get paid for $1 bet)
  - Example: 2.86 means $1 bet returns $2.86

- **EV %**: Expected value percentage
  - Positive = Value (bet profitable long-term)
  - Negative = Bad bet (avoid)
  - Formula: EV = (Prob × Odds) - 1

#### Using Combo Opportunities

1. **Sort by EV %**: Highest EV first
2. **Filter by Recommendation**: BET > MONITOR > PASS
3. **Check Odds**: Compare to your bookmaker
4. **Verify Markets**: Confirm bookmaker offers the combo
5. **Calculate Stake**: Use Kelly Criterion or flat bet
6. **Track Results**: Log outcomes for analysis

---

### Data Export

#### JSON Download
- Click "⬇️ Download current results as JSON"
- Includes:
  - All matches and picks
  - Combo opportunities
  - ML predictions (if available)
  - Metadata and filters applied

#### CSV Export
- Corner predictions tab offers CSV download
- Includes:
  - Match details
  - Predictions
  - Market lines
  - Recommendations

---

## Advanced Usage

### Configuration

#### Edit train_config.yaml
```yaml
ml_config:
  mode: predict
  algorithms:
    - xgboost
    - random_forest
  
  train:
    min_samples: 300
    decay_factor: 0.85
    
  predict:
    confidence_threshold: 0.5
    validate_signature: true
```

#### Customize Analysis

```bash
# High confidence, double chance enabled
python3 cli.py full-league --league E0 \
  --min-confidence 0.70 \
  --enable-double-chance \
  --dc-min-prob 0.75

# Low confidence, combo focus
python3 cli.py full-league --league E0 \
  --min-confidence 0.50

# ML training run
python3 cli.py full-league --league E0 \
  --ml-mode train \
  --min-samples 400
```

### Troubleshooting

#### Streamlit Won't Load Data
```bash
# Check if files exist
ls -la data/analysis/full_league_suggestions_*.json | head -5

# Check file timestamps
ls -lt data/analysis/consolidated_full_league_*.json | head -1

# Verify JSON format
python3 -m json.tool data/analysis/consolidated_full_league_*.json > /dev/null
```

#### No Combos Appearing
```bash
# Check if combos exist in JSON
grep -c "combo_picks" data/analysis/consolidated_full_league_*.json

# Debug analysis
python3 cli.py full-league --league E0 --ml-mode off 2>&1 | grep -i combo
```

#### Memory Issues
```bash
# Run single league at a time
python3 cli.py full-league --league E0

# Reduce match history
python3 cli.py full-league --league E0 --rating-last-n 4
```

---

## Workflow Summary

✅ **Complete automated football analytics pipeline**

1. **Input**: Fixtures and historical data
2. **Process**: xG, markets, combos, ML predictions
3. **Output**: JSON files, text reports
4. **Display**: Interactive Streamlit dashboard
5. **Export**: Download data as JSON/CSV

### Key Features
- ✅ Poisson analysis (always)
- ✅ Combo markets (multi-leg combinations)
- ✅ ML predictions (optional)
- ✅ Double Chance markets (optional)
- ✅ Cross-league parlays (automatic)
- ✅ Streamlit visualization (interactive)

### Ready to Use
All components integrated and validated. Run any analysis command above to get started.

