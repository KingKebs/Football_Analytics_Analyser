# Football Analytics Analyser - Quick Reference Guide

**Last Updated:** November 13, 2025

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
# See the latest analysis
python cli.py --task view

# Run backtest on results
python cli.py --task backtest
```

---

## 📊 Common Tasks

### Analyze English Premier League (EPL)
```bash
python cli.py --task full-league --league E0 --rating-model blended --last-n 6
```

### Analyze Multiple Leagues
```bash
python cli.py --task download --leagues E0,SP1,D1,F1 --season AUTO
python cli.py --task full-league --league E0,SP1,D1
```

### Analyze Single Match
```bash
python cli.py --task single-match --home Arsenal --away Chelsea
```

### Analyze Corners in Dataset
```bash
python cli.py --task corners --file E0_2425.csv
python cli.py --task analyze-corners  # All data
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

### Rating Models
- `blended` - Combines recent form with baseline ratings (recommended)
- `poisson` - Poisson distribution based on shot data
- `xg` - Expected goals model

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
├── corners_analysis.py             # Corner pattern analyzer
├── algorithms.py                   # Core algorithms & models
├── automate_football_analytics_fullLeague.py  # League analysis
├── automate_football_analytics.py  # Single match analysis
│
├── data/                           # Output & analysis results
│   ├── analysis/                   # Full league suggestions (JSON)
│   ├── raw/                        # Original CSVs
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
│   └── cli_20251113_*.log
│
├── tests/                          # Test files
└── requirements.txt                # Dependencies
```

---

## 🎯 Typical Workflows

### Workflow 1: Daily Analysis
```bash
# Morning: Download latest data
python cli.py --task download --leagues E0,SP1 --season AUTO

# Validate
python cli.py --task validate --check-all

# Analyze EPL
python cli.py --task full-league --league E0

# View suggestions
python cli.py --task view --file data/analysis/full_league_suggestions_E0_*.json
```

### Workflow 2: Full League Study
```bash
# Download all major leagues
python cli.py --task download --leagues E0,SP1,D1,F1,I1 --season AUTO

# Organize
python cli.py --task organize

# Validate
python cli.py --task validate --check-all

# Analyze each league
python cli.py --task full-league --league E0
python cli.py --task full-league --league SP1
python cli.py --task full-league --league D1

# Corner analysis
python cli.py --task analyze-corners

# Backtest all results
python cli.py --task backtest
```

### Workflow 3: Cleanup & Maintenance
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
- **Contains:** Match predictions, odds, betting suggestions

### Team Statistics
- **Location:** `data/team_stats/`
- **Format:** `home_away_team_strengths_<LEAGUE>.csv`
- **Contains:** Offensive/defensive ratings, win rates

### Corner Analysis
- **Location:** `data/`
- **Format:** `corners_analysis_<TIMESTAMP>.csv`
- **Contains:** Corner predictions, half-split estimates

### Data Manifest
- **Location:** `data/MANIFEST.json`
- **Contains:** File inventory, sizes, modification dates

---

## 🔍 Troubleshooting

### "No CSV files found"
```bash
# Download data first
python cli.py --task download --leagues E0 --season AUTO
```

### "Team not found"
```bash
# Verify exact team name in league data
python cli.py --task view --file data/league_data_E0.csv

# Use exact case-sensitive name
python cli.py --task single-match --home "Manchester United" --away "Liverpool"
```

### "JSON decode error"
```bash
# Validate and fix JSON
python data_manager.py --validate

# Regenerate analysis if corrupted
python cli.py --task full-league --league E0
```

### Script won't execute
```bash
# Check logs
tail -f logs/cli_*.log

# Run with verbose output
python cli.py --task full-league --league E0 --verbose
```

---

## 🛠️ Customization

### Change Default Model
Edit `cli.py` line ~50 and change:
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
3. **Cache results** - Check `/data/cache/` for reusable models
4. **Parallel analysis** - Run multiple league analyses in separate terminals:
   ```bash
   # Terminal 1
   python cli.py --task full-league --league E0 &
   
   # Terminal 2
   python cli.py --task full-league --league SP1 &
   ```

---

## 📞 Getting Help

```bash
# Show this guide
python cli.py --task help

# Show CLI options
python cli.py --help

# Show data manager options
python data_manager.py --help

# Show corners analyzer help
python corners_analysis.py --help

# Check logs for errors
tail -20 logs/cli_*.log
```

---

## 🎓 Next Steps

1. **Run your first analysis:** `python cli.py --task full-league --league E0`
2. **View results:** `python cli.py --task view`
3. **Analyze corners:** `python cli.py --task analyze-corners`
4. **Backtest:** `python cli.py --task backtest`
5. **Read PROJECT_STRUCTURE_ANALYSIS.md** for detailed architecture

---

*For more information, see PROJECT_STRUCTURE_ANALYSIS.md and README.md*

