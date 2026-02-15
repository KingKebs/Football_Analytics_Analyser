# Dynamic Analysis Workflow

## Overview

The `run_analysis_workflow.py` script provides an automated, interactive workflow for football analytics that:

1. ✅ Reads upcoming matches from `data/raw/upcomingMatches.json`
2. ✅ Auto-detects leagues and prompts for confirmation
3. ✅ Converts fixtures to parsed format
4. ✅ Runs full-league analysis with ML predictions (preserves fixed feature engineering)
5. ✅ Runs corners analysis with ML predictions
6. ✅ Generates consolidated outputs for Streamlit UI

## Quick Start

```bash
# Interactive mode (recommended for first-time use)
python run_analysis_workflow.py --date 2026-02-11

# Auto mode (no prompts, uses detected leagues)
python run_analysis_workflow.py --date 2026-02-11 --auto

# Override league detection
python run_analysis_workflow.py --date 2026-02-11 --leagues E0,E2,E3

# Use today's date
python run_analysis_workflow.py --auto
```

## Workflow Steps

### Step 1: Read Upcoming Matches
- Loads `data/raw/upcomingMatches.json`
- Validates file exists and is readable

### Step 2: Auto-Detect Leagues
- Scans upcoming matches for competition names
- Maps competitions to league codes (E0, E1, E2, SP1, etc.)
- Displays detected leagues and match count

### Step 3: Confirm Leagues
- Shows detected leagues for user confirmation
- Prompts: "Proceed with these N league(s)? (y/n)"
- Skip prompt in `--auto` mode
- Override with `--leagues` flag

### Step 4: Convert Fixtures
- Runs: `python cli.py --task convert-upcoming`
- Converts to `todays_fixtures_<DATE>.json` in `data/analysis/`
- Preserves dates from upcoming matches

### Step 5: Full-League Analysis
- Runs: `python cli.py --task full-league --use-parsed-all --ml-mode predict`
- Auto-detects leagues from parsed fixtures
- Generates per-league suggestions: `data/analysis/full_league_suggestions_<LEAGUE>_<DATE>.json`
- Generates consolidated output: `data/analysis/consolidated_full_league_<DATE>.json`
- **ML predictions use fixed feature engineering** (no identical predictions bug)

### Step 6: Corners Analysis
- Runs: `python cli.py --task corners --use-parsed-all --corners-use-ml-prediction`
- Auto-detects leagues from parsed fixtures
- Generates: `data/corners/parsed_corners_predictions_<DATE>.json`
- Uses team aliases for robust matching

## Outputs

All outputs are stored in standard locations for Streamlit UI consumption:

### Full-League Outputs
```
data/analysis/
├── full_league_suggestions_E0_20260211_*.json    # Per-league
├── full_league_suggestions_E2_20260211_*.json
├── full_league_suggestions_E3_20260211_*.json
└── consolidated_full_league_20260211_*.json      # All leagues combined
```

### Corners Outputs
```
data/corners/
└── parsed_corners_predictions_20260211.json      # All leagues
```

## Command-Line Options

```
--date YYYY-MM-DD     Analysis date (default: today)
--leagues E0,E2,E3    Override auto-detected leagues
--auto                Skip all confirmation prompts
--verbose             Enable verbose output for debugging
```

## Examples

### Daily Workflow
```bash
# 1. Update upcoming matches (manual or via API)
# Edit data/raw/upcomingMatches.json

# 2. Run workflow with prompts
python run_analysis_workflow.py --date 2026-02-11

# 3. View results in Streamlit
streamlit run src/streamlit_app.py
```

### Automated/Cron Job
```bash
# Run without prompts for automation
python run_analysis_workflow.py --auto --verbose >> logs/workflow_$(date +%Y%m%d).log 2>&1
```

### Specific Leagues Only
```bash
# Analyze only Premier League and League One
python run_analysis_workflow.py --date 2026-02-11 --leagues E0,E2
```

## Integration with Existing Tools

### Compatible with Manual Commands
The workflow script uses the same underlying CLI commands, so outputs are 100% compatible:

```bash
# Manual step-by-step (what the workflow does internally):
python cli.py --task convert-upcoming --file data/raw/upcomingMatches.json --output-dir data/analysis --date 2026-02-11
python cli.py --task full-league --use-parsed-all --fixtures-date 20260211 --ml-mode predict --enable-double-chance --verbose
python cli.py --task corners --use-parsed-all --fixtures-date 20260211 --league ALL --min-team-matches 3 --corners-use-ml-prediction --verbose
```

### Streamlit UI
The workflow outputs are automatically discovered by Streamlit:
- Latest consolidated file shown by default
- Per-league files available in dropdown
- Corners predictions displayed in dedicated tab

## Troubleshooting

### "No leagues detected"
- Check `data/raw/upcomingMatches.json` exists and has valid JSON
- Ensure competition names match LEAGUE_MAP (Premier League, Championship, etc.)
- Use `--leagues` to manually specify

### "Analysis failed"
- Check logs in `logs/cli_*.log`
- Run with `--verbose` for detailed output
- Verify historical data downloaded: `python cli.py --task download --leagues E0,E2,E3 --season AUTO`

### "ML predictions identical"
- This was fixed in the feature engineering pipeline
- Workflow preserves the fix by using `--ml-mode predict` correctly
- Check git logs for "ML predictions fix" commits

## Maintenance

### Update League Mappings
Edit `LEAGUE_MAP` in `run_analysis_workflow.py` to add new competitions:

```python
LEAGUE_MAP = {
    'Premier League': 'E0',
    'Your New Competition': 'XYZ',
    # ...
}
```

### Adjust Default Parameters
Modify the command-building methods in the workflow class:
- `run_full_league_analysis()` for full-league params
- `run_corners_analysis()` for corners params

## Quick Reference Commands

See `docs/quick_commands.py` for optimized parameter configurations:

```bash
python docs/quick_commands.py
```

## License

Same as parent project.

