# Football Analytics Analyser - Quick Reference Guide

**Last Updated:** December 6, 2025

## 🆕 **NEW FEATURE: Dynamic League Extraction**
When using `--use-parsed-all` with `--leagues ALL` (or no `--leagues` specified), the system now automatically detects and processes only the leagues present in your parsed fixtures file. This eliminates the "ALL.csv not found" error and makes the system much more resource-efficient.

**Example:**
```bash
# Before: Would fail with "ALL.csv not found" 
# After: Automatically processes E0,F1,D1,I1,N1,SP1 based on fixtures content
python cli.py --task full-league --use-parsed-all --fixtures-date 20251206 --min-confidence 0.6 --ml-mode predict --enable-double-chance --verbose
```

---

## 🚀 Quick Start

### First Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download data for a league
python cli.py --task download --leagues E0 --season AUTO

# Validate downloaded data
python cli.py --task validate --check-all

# Analyze the league
python cli.py --task full-league --league E0
```

### View Results
```bash
# See the latest analysis in Streamlit UI
streamlit run src/streamlit_app.py

# Or view via CLI
python cli.py --task view --file data/analysis/full_league_suggestions_E0_*.json
```

---

## 📊 Common Tasks

### Analyze English Premier League (EPL)
```bash
python cli.py --task full-league --league E0 --rating-model blended --last-n 6
```

### Analyze Multiple Leagues
```bash
# Manual approach - specify leagues explicitly
python cli.py --task download --leagues E0,SP1,D1,F1 --season AUTO
python cli.py --task full-league --league E0,SP1,D1

# Dynamic approach - automatically detect leagues from parsed fixtures
python cli.py --task full-league --use-parsed-all --min-confidence 0.6 --ml-mode predict --verbose
```

### Full League Analysis with Dynamic League Detection
```bash
# Automatically detects and processes leagues from parsed fixtures
python cli.py --task full-league --use-parsed-all --min-confidence 0.6 --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --verbose

# Process specific date's fixtures (auto-detects leagues)
python cli.py --task full-league --use-parsed-all --fixtures-date 20251206 --min-confidence 0.6 --ml-mode predict --enable-double-chance --verbose

# Traditional approach - specify leagues manually
python cli.py --task full-league --leagues E0,SP1,D1 --min-confidence 0.6 --ml-mode predict --verbose
```

### Analyze Single Match
```bash
python cli.py --task single-match --home Arsenal --away Chelsea
```

### Corner Analysis

#### Full Corner Analysis on All Data
```bash
python cli.py --task analyze-corners --enable-double-chance --dc-min-prob 0.75 --ml-mode predict --verbose
```

#### Corner Analysis on Specific File
```bash
python cli.py --task corners --file E0_2425.csv --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple --ml-mode predict --verbose
```

#### Parsed Fixture Corner Predictions (Dynamic League Discovery)
```bash
# Automatically detects leagues present in today's parsed fixtures
python cli.py --task corners --use-parsed-all --enable-double-chance --dc-min-prob 0.75 --ml-mode predict --verbose
```

#### Parsed Fixture Corner Predictions (Specific Date)
```bash
# Automatically processes only leagues found in the specified date's fixtures
python cli.py --task corners --use-parsed-all --fixtures-date 20251205 --min-team-matches 5 --enable-double-chance --dc-min-prob 0.75 --ml-mode predict --verbose
```

#### Single Match Corner Projection
```bash
python cli.py --task corners --home-team Arsenal --away-team Chelsea --enable-double-chance --dc-min-prob 0.75 --ml-mode predict --verbose
```

### Manage Data
```bash
# Generate manifest of all files
python data_manager.py --manifest

# List files by league
python data_manager.py --list-leagues

# Archive files older than 7 days
python data_manager.py --archive --days 7

# Full cleanup (manifest + archive + validate)
python data_manager.py --full-cleanup --dry-run  # Preview first
python data_manager.py --full-cleanup             # Execute
```

---

## 🔧 Detailed CLI Options

### Double Chance Analysis
- `--enable-double-chance` - Enable double chance market analysis (1X, X2, 12)
- `--dc-min-prob <float>` - Minimum probability threshold for double chance (e.g., 0.75)
- `--dc-secondary-threshold <float>` - Secondary threshold for additional double chance picks (e.g., 0.80)
- `--dc-allow-multiple` - Allow multiple double chance suggestions per match

### Corner Analysis Flags
- `--use-parsed-all` - Use parsed fixtures (todays_fixtures_*.csv/json) with **dynamic league detection** - automatically processes only leagues found in fixtures
- `--fixtures-date <YYYYMMDD>` - Specify date for parsed fixtures file (searches both `data/` and `data/analysis/`)
- `--min-team-matches <int>` - Minimum historical matches per team for corner prediction (default: 5)
- `--home-team <name>` - Specific home team for corner prediction
- `--away-team <name>` - Specific away team for corner prediction
- `--corners-use-ml-prediction` - Use ML models with confidence ranges for corner predictions
- `--corners-mc-samples <int>` - Monte Carlo samples for corner range estimation (default: 1000)
- `--save-enriched` - Save enriched engineered corner feature CSVs

### Rating Models
- `blended` - Combines recent form with baseline ratings (recommended)
- `poisson` - Poisson distribution based on shot data
- `xg` - Expected goals model

### ML Mode
- `--ml-mode <mode>` - Machine learning mode for predictions. Options: `predict`, `train`, `off` (default: `off`)
- `--ml-validate` - Show cross-validation metrics for ML models
- `--ml-save-models` - Persist trained ML models to disk

### Verbose Output
- `--verbose` - Increase output verbosity (detailed logging and summaries)

### Blend Weight
- `0.0` - Ignore form, use only baseline ratings
- `0.3` - 30% recent form, 70% baseline (default, recommended)
- `0.7` - 70% recent form, 30% baseline (aggressive form tracking)

### Last N Matches
- `4` - Very recent form only
- `6` - Standard (default)
- `10` - Long-term trends

### League Codes
| Code | League | Country |
|------|--------|---------|
| E0 | Premier League | England |
| E1 | Championship | England |
| E2 | League One | England |
| E3 | League Two | England |
| SP1 | La Liga | Spain |
| D1 | Bundesliga | Germany |
| F1 | Ligue 1 | France |
| I1 | Serie A | Italy |
| B1 | Pro League | Belgium |
| P1 | Primeira Liga | Portugal |
| SC0 | Scottish Premier | Scotland |
| N1 | Eredivisie | Netherlands |

---

## 📁 Directory Structure

```
├── cli.py                          # Unified CLI entry point
├── data_manager.py                 # Data organization tool
├── requirements.txt                # Dependencies
│
├── src/
│   ├── streamlit_app.py            # Interactive Streamlit dashboard
│   ├── corners_analysis.py         # Corner pattern analyzer
│   ├── algorithms.py               # Core algorithms & models
│   ├── automate_football_analytics_fullLeague.py  # League analysis
│   ├── automate_football_analytics.py  # Single match analysis
│   ├── convert_upcoming_matches.py # Convert upcoming games JSON
│   └── ...                         # Other analysis modules
│
├── data/                           # Output & analysis results
│   ├── analysis/                   # Full league suggestions (JSON)
│   ├── raw/                        # Original CSVs & fixtures
│   ├── corners/                    # Corner analysis results
│   ├── team_stats/                 # Team strength matrices
│   ├── fixtures/                   # Match fixtures
│   ├── archive/                    # Old files (>7 days)
│   └── MANIFEST.json               # File inventory
│
├── football-data/                  # Downloaded league data
│   ├── E0_2425.csv
│   ├── SP1_2425.csv
│   └── ...
│
├── logs/                           # Execution logs
│   └── cli_20251205_*.log
│
├── models/                         # Trained ML models
│   └── corners/                    # Corner prediction models
│
├── tests/                          # Test files
└── ReadMeDocs/                     # Documentation
    └── QUICK_REFERENCE.md          # This file
```

---

## 🎯 Typical Workflows

### Workflow 1: Daily Analysis (Modern Approach)
```bash
# Option A: Using parsed fixtures (RECOMMENDED - most efficient)
# If you have today's fixtures parsed:
python cli.py --task full-league --use-parsed-all --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple --verbose

# Option B: Traditional approach
# Download latest data first
python cli.py --task download --leagues E0,SP1 --season AUTO
python cli.py --task validate --check-all
python cli.py --task full-league --league E0 --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple --verbose

# View suggestions in Streamlit
streamlit run src/streamlit_app.py
```

### Workflow 2: Full League Study
```bash
# Method 1: Traditional approach - Download all major leagues
python cli.py --task download --leagues E0,SP1,D1,F1,I1 --season AUTO
python cli.py --task organize
python cli.py --task validate --check-all

# Analyze each league individually
python cli.py --task full-league --league E0 --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple --verbose
python cli.py --task full-league --league SP1 --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple --verbose

# Method 2: Dynamic approach using parsed fixtures (RECOMMENDED)
# If you have parsed fixtures, this automatically processes all relevant leagues
python cli.py --task full-league --use-parsed-all --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple --verbose

# Full corner analysis with ML predictions (dynamic detection)
python cli.py --task corners --use-parsed-all --enable-double-chance --dc-min-prob 0.75 --ml-mode predict --corners-use-ml-prediction --verbose

# Backtest all results
python cli.py --task backtest
```

### Workflow 3: Convert Upcoming Games and Analyze
```bash
# Download today's fixtures from free APIs
python cli.py --task download-fixtures --update-today

# Convert upcoming games from JSON to CSV/JSON format
python cli.py --task convert-upcoming --file data/raw/upcomingMatches.json --output-dir data/analysis

# Output creates: data/analysis/todays_fixtures_YYYYMMDD.csv and .json

# Validate the converted fixtures
python cli.py --task validate --file data/analysis/todays_fixtures_*.csv

# Analyze the upcoming games (dynamically detects leagues from fixtures)
python cli.py --task full-league --use-parsed-all --min-confidence 0.6 --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple --verbose

# View the upcoming fixtures analysis
streamlit run src/streamlit_app.py
```

### Workflow 4: Corner Analysis on Upcoming Games
```bash
# Download and convert upcoming fixtures
python cli.py --task download-fixtures --update-today
python cli.py --task convert-upcoming --file data/raw/upcomingMatches.json --output-dir data/analysis

# Get corner predictions for today's games (processes leagues found in fixtures)
python cli.py --task corners --use-parsed-all --min-team-matches 5 --enable-double-chance --dc-min-prob 0.75 --ml-mode predict --corners-use-ml-prediction --verbose

# View results
streamlit run src/streamlit_app.py
```

### Workflow 5: Cleanup & Maintenance
```bash
# Preview what will be archived
python data_manager.py --full-cleanup --dry-run

# Validate JSON
python data_manager.py --validate

# Generate manifest
python data_manager.py --manifest

# Execute cleanup
python data_manager.py --full-cleanup
```

---

## 📊 Output Files

### Analysis Results
- **Location:** `data/analysis/`
- **Format:** `full_league_suggestions_<LEAGUE>_<TIMESTAMP>.json`
- **Contains:** Match predictions, odds, betting suggestions, parlay analysis
- **Dynamic Processing:** When using `--use-parsed-all`, generates one file per league found in fixtures

### Corner Analysis Results
- **Location:** `data/corners/`
- **Format:** `parsed_corners_predictions_<DATE>.json` (when using `--use-parsed-all`)
- **Format:** `corners_analysis_<LEAGUE>_<TIMESTAMP>.csv` / `.json` (league-specific)
- **Contains:** Corner predictions, half-split estimates, team statistics
- **Dynamic Processing:** Processes all leagues found in parsed fixtures automatically

### Upcoming Fixtures
- **Location:** `data/analysis/`
- **Format:** `todays_fixtures_YYYYMMDD.csv` / `.json`
- **Contains:** Converted upcoming games ready for analysis

### Team Statistics
- **Location:** `data/team_stats/`
- **Format:** `home_away_team_strengths_<LEAGUE>.csv`
- **Contains:** Offensive/defensive ratings, win rates

### Data Manifest
- **Location:** `data/MANIFEST.json`
- **Contains:** File inventory, sizes, modification dates

---

## 🎨 Viewing Results

### Streamlit Dashboard (Recommended)
```bash
streamlit run src/streamlit_app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false
```
Features:
- Interactive league suggestions with filtering
- Corner analysis visualizations
- Team statistics comparisons
- Parlay analysis
- Live corner predictions

### CLI Viewer
```bash
python cli.py --task view --file data/analysis/full_league_suggestions_E0_*.json
```

### JSON/CSV Export
```bash
python cli.py --task full-league --league E0 --output-format json
python cli.py --task full-league --league E0 --output-format csv
```

---

## 🔍 Troubleshooting

### "No full league suggestion files found in data/analysis/"
```bash
# Run an analysis first
python cli.py --task full-league --league E0

# Then view
streamlit run src/streamlit_app.py
```

### "No CSV files found"
```bash
# Download data first
python cli.py --task download --leagues E0 --season AUTO
```

### "Team not found"
```bash
# Verify exact team name in league data
python cli.py --task corners --league E0 --top-n 10

# Use exact case-sensitive name
python cli.py --task corners --home-team "Manchester United" --away-team "Liverpool"
```

### "CornersAnalyzer object has no attribute 'run_full_analysis'"
```bash
# This issue is fixed in the latest version
# Update your cli.py to the latest version
python cli.py --task analyze-corners --enable-double-chance --dc-min-prob 0.75 --ml-mode predict
```

### "League CSV not found: football-data/all-euro-football/ALL.csv"
```bash
# This error occurs when using --leagues ALL without parsed fixtures
# SOLUTION 1: Use dynamic league detection with parsed fixtures
python cli.py --task full-league --use-parsed-all --min-confidence 0.6 --ml-mode predict --verbose

# SOLUTION 2: Specify leagues explicitly
python cli.py --task full-league --leagues E0,SP1,D1 --min-confidence 0.6 --ml-mode predict --verbose

# SOLUTION 3: Check if you have parsed fixtures file
ls data/analysis/todays_fixtures_*.json
```

### "No leagues found in parsed fixtures, falling back to default"
```bash
# This means the parsed fixtures file is empty or has no valid league data
# Check the fixtures file content:
python -c "import json; print(json.load(open('data/analysis/todays_fixtures_20251206.json'))[:2])"

# Ensure the fixtures file has 'league' or 'League' field populated
# If using manual league specification instead:
python cli.py --task full-league --leagues E0,SP1 --min-confidence 0.6 --verbose
```

### Streamlit app shows "No full league suggestion files found"
```bash
# Run an analysis to generate results
python cli.py --task full-league --league E0

# The app searches in data/analysis/ by default
# Results are displayed in the Streamlit UI at http://localhost:8501
```

### Script won't execute
```bash
# Check logs
tail -f logs/cli_*.log

# Run with verbose output
python cli.py --task full-league --league E0 --verbose
```

---

## ✨ Best Practices

### Use Dynamic League Detection (Recommended)
```bash
# ✅ BEST: Automatically processes only leagues with actual fixtures
python cli.py --task full-league --use-parsed-all --min-confidence 0.6 --ml-mode predict --verbose

# ✅ BEST: Corner analysis with dynamic detection
python cli.py --task corners --use-parsed-all --min-team-matches 5 --corners-use-ml-prediction --verbose

# ❌ AVOID: Manual specification when you have parsed fixtures (wastes resources)
python cli.py --task full-league --leagues E0,SP1,D1,F1,I1 --min-confidence 0.6 --verbose
```

### Resource Efficiency
- **Dynamic detection** only processes leagues with actual fixtures (saves time and resources)
- **Automatic fallback** to default league (E0) if no fixtures found
- **Multi-directory search** finds fixtures in both `data/` and `data/analysis/`

### File Organization
```bash
# Place parsed fixtures in data/analysis/ (recommended)
data/analysis/todays_fixtures_20251206.json

# Or data/ directory (also supported)  
data/todays_fixtures_20251206.json

# System searches both locations automatically
```

---

## 🛠️ Customization

### Change Default Model
Edit `cli.py` line ~150 and change:
```python
default='blended'  # Change to 'poisson' or 'xg'
```

### Adjust Logging Level
```bash
# More verbose
python cli.py --task full-league --league E0 --verbose

# Check logs
cat logs/cli_*.log
```

### Extend with New Tasks

---

## 🔧 Technical Details: Dynamic League Extraction

### How It Works
The dynamic league extraction feature automatically identifies which leagues have fixtures in your parsed data and processes only those leagues. This eliminates resource waste and prevents errors.

### Implementation Details
```python
# Simplified implementation flow:
1. Load parsed fixtures file (searches data/ and data/analysis/)
2. Extract unique league codes from 'League' column
3. Filter to only supported leagues (E0, SP1, D1, etc.)
4. Process each league individually
5. Generate separate output files per league
```

### Supported League Detection
- **Primary field:** `League` column in fixtures file
- **Fallback field:** `Competition` column (with mapping)
- **Supported codes:** E0, E1, E2, E3, D1, D2, SP1, SP2, I1, I2, F1, F2, N1, P1, SC0-SC3, B1, G1, T1
- **Auto-mapping:** "Premier League" → E0, "La Liga" → SP1, etc.

### File Search Priority
1. `data/analysis/todays_fixtures_<DATE>.json`
2. `data/analysis/todays_fixtures_<DATE>.csv`
3. `data/todays_fixtures_<DATE>.json`
4. `data/todays_fixtures_<DATE>.csv`
5. Most recent `data/analysis/todays_fixtures_*.json`
6. Most recent `data/todays_fixtures_*.json`

### Example Output
```bash
# Input fixtures contain: E0, SP1, D1 matches
# Command:
python cli.py --task full-league --use-parsed-all --verbose

# Output:
[INFO] Extracted 3 leagues from parsed fixtures: E0,SP1,D1
[INFO] Dynamically using leagues from parsed fixtures: E0,SP1,D1
# Generates:
data/analysis/full_league_suggestions_E0_20251206_123456.json
data/analysis/full_league_suggestions_SP1_20251206_123457.json  
data/analysis/full_league_suggestions_D1_20251206_123458.json
```

### Error Handling
- **No fixtures found:** Falls back to E0 (Premier League)
- **Empty fixtures:** Logs warning and uses default
- **Invalid league codes:** Skips unknown leagues, processes valid ones
- **Missing league data:** Skips leagues without historical data

---

## 📅 Version History

### v2.1.0 (December 6, 2025)
- ✅ **NEW:** Dynamic League Extraction for `--use-parsed-all`
- ✅ **IMPROVED:** Multi-directory fixtures file search
- ✅ **FIXED:** "ALL.csv not found" error when using parsed fixtures
- ✅ **ENHANCED:** Resource efficiency - only processes leagues with fixtures
- ✅ **ADDED:** Automatic league code mapping from competition names

### v2.0.0 (December 5, 2025)
- ✅ Enhanced corner analysis with ML predictions
- ✅ Double chance market analysis
- ✅ Improved Streamlit dashboard
- ✅ Better error handling and logging

---

*For more detailed documentation, see the full README.md and other files in ReadMeDocs/*
Edit `cli.py`, add method to `FootballAnalyticsCLI` class:
```python
def task_my_analysis(self, args):
    """My custom analysis."""
    from my_module import analyze
    analyze()
    return 0
```

Then add to task choices and call in `run()` method.

---

## 📈 Performance Tips

1. **First run is slowest** - Download + organize takes time
2. **Use specific leagues** - E0 is faster than all leagues
3. **Cache results** - Check `data/corners/` for reusable models
4. **Parallel analysis** - Run multiple league analyses in separate terminals:
   ```bash
   # Terminal 1
   python cli.py --task full-league --league E0 &
   
   # Terminal 2
   python cli.py --task full-league --league SP1 &
   ```
5. **Use ML predictions** - Once trained, ML models are fast:
   ```bash
   python cli.py --task full-league --league E0 --ml-mode predict --ml-save-models
   ```

---

## 📞 Getting Help

```bash
# Show comprehensive help
python cli.py --task help

# Show all CLI options
python cli.py --help

# Show data manager options
python data_manager.py --help

# Show corners analyzer options
python src/corners_analysis.py --help

# Check recent logs
tail -20 logs/cli_*.log

# View Streamlit app info
streamlit run src/streamlit_app.py --logger.level=debug
```

---

## 🎓 Next Steps

1. **Run your first analysis:** `python cli.py --task full-league --league E0`
2. **View results in Streamlit:** `streamlit run src/streamlit_app.py`
3. **Analyze corners:** `python cli.py --task analyze-corners`
4. **Download upcoming fixtures:** `python cli.py --task download-fixtures --update-today`
5. **Convert and analyze:** `python cli.py --task convert-upcoming --file data/raw/upcomingMatches.json --output-dir data/analysis`
6. **Read PROJECT_STRUCTURE_ANALYSIS.md** for detailed architecture

---

## ✨ Recent Updates (December 5, 2025)

- ✅ Fixed `analyze-corners` task - now uses proper workflow sequence
- ✅ Added Streamlit dashboard for interactive result viewing
- ✅ Improved corner analysis with ML predictions support
- ✅ Added `convert-upcoming` task for converting upcoming games JSON
- ✅ Updated all workflows with latest commands and options
- ✅ Fixed line-by-line table display for match suggestions
- ✅ Added upcoming fixtures workflow examples

*For more information, see PROJECT_STRUCTURE_ANALYSIS.md and README.md*
