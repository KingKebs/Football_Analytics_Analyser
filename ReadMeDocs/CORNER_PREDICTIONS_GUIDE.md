# Corner Predictions Workflow Guide (Updated Nov 2024)

## 🚀 What's New: Steps 1-4 Improvements Applied!

This workflow now includes advanced modeling improvements:
- ✅ **Step 1**: Cross-validation for honest performance metrics
- ✅ **Step 2**: Ensemble models (RandomForest, XGBoost)
- ✅ **Step 3**: Recency weighting (recent matches weighted higher)
- ✅ **Step 4**: Interaction features (shots×fouls, goal-state×pressure, etc.)

**Result:** More accurate predictions with ~2.3-2.5 corners MAE (vs ~2.7 previously)

---

## 🎯 Quick Start: Today's Matches

### Clean Data Directory (Recommended First Step)
```bash
# Preview what will be organized (dry run)
python3 organize_structure.py --dry-run

# Organize the data directory
python3 organize_structure.py
```

This organizes files into:
- `data/corners/` - Corner analysis and predictions (89 files)
- `data/league_analysis/` - Full league suggestions (25 files)
- `data/fixtures/` - Parsed match fixtures (7 files)
- `data/archived/` - Old files >7 days (11 files)
- `data/*.csv` - League data tables (kept in root)

## Workflow Steps

### 1. Parse Match Log → Extract Fixtures
```bash
python3 parse_match_log.py --input tmp/corners/251115_match_games.log --league-code E2
```
- Reads raw match log
- Extracts scheduled fixtures (ignores postponed by default)
- Outputs: `data/todays_fixtures_YYYYMMDD.csv` and `.json`

### 2. Run Corners Analysis → Build Team Stats
```bash
python3 corners_analysis.py --league E2 --no-prompt
```
- Aggregates historical match data for E2
- Computes team corner statistics (home/away)
- Calculates half-split ratios
- Outputs: `data/corners/team_stats_E2_*.json`

### 3. Generate Predictions → For Each Fixture
```bash
python3 corners_analysis.py --league E2 --home-team Burton --away-team Blackpool
```
- Predicts total corners (mean + range)
- Predicts 1H vs 2H split
- Suggests betting lines with probabilities
- Outputs: `data/corners/match_prediction_E2_Burton_vs_Blackpool_*.json`

## Automated Workflow (All-in-One)

### Basic Usage
```bash
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2
```
- Prompts you for each fixture (Y/n/q)
- Generates predictions interactively

### Auto Mode (No Prompts)
```bash
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2 --auto
```
- Processes all fixtures automatically
- Exports batch predictions JSON

### With Season Filter
```bash
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2 --seasons 2425 --auto
```
- Uses only 2024/25 season data

### Force Re-Analysis
```bash
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2 --auto --force
```
- Re-runs corners analysis even if done today

## Output Files

### Parsed Fixtures
- **Location**: `data/todays_fixtures_YYYYMMDD.csv` / `.json`
- **Contents**: Date, league, time, home team, away team, status

### Team Statistics
- **Location**: `data/corners/team_stats_E2_YYYYMMDD_HHMMSS.json`
- **Contents**: Home/away corner averages, std dev, matches played

### Individual Predictions
- **Location**: `data/corners/match_prediction_E2_Home_vs_Away_YYYYMMDD_HHMMSS.json`
- **Contents**: Corner means, ranges, 1H/2H split, suggested lines with probabilities

### Batch Predictions
- **Location**: `data/corners/batch_predictions_E2_YYYYMMDD_HHMMSS.json`
- **Contents**: All predictions for the day in one file

## Example Output

For **Burton vs Blackpool**:
```
Home Mean: 4.45 (Std≈2.03)  
Away Mean: 4.58 (Std≈2.9)
Total Mean: 9.04 Range: 6.57 - 11.5
1H Mean: 3.55  2H Mean: 5.49 (Ratio1H=0.39)

Suggested Total Corner Lines:
  7.5: Over 72.7% / Under 27.3% => OVER
  10.5: Over 26.9% / Under 73.1% => UNDER
  11.5: Over 15.4% / Under 84.6% => UNDER

Suggested 1H Corner Lines:
  3.5: Over 49.9% / Under 50.1%
  4.5: Over 24.9% / Under 75.1% => UNDER
```

## Tips

1. **First Time**: Run with `--force` to ensure fresh analysis
2. **Multiple Leagues**: Run separate commands for each league code
3. **Season-Specific**: Use `--seasons 2425` to focus on current season only
4. **Quick Review**: Check `batch_predictions_*.json` for all results in one file
5. **Re-run Safely**: Script checks if analysis was done today (skips if yes, unless `--force`)

## Common League Codes

- **E0**: Premier League
- **E1**: Championship
- **E2**: League One
- **E3**: League Two
- **D1**: Bundesliga
- **SP1**: La Liga
- **I1**: Serie A
- **F1**: Ligue 1

## Troubleshooting

### "Home team not found"
- Team name mismatch between log and historical data
- Script uses fuzzy matching (default)
- Check actual team names: `python3 corners_analysis.py --league E2 --list-teams`

### "No CSV files found"
- Need to download data first: `python3 cli.py --task download --leagues E2`

### "Analysis already run today"
- Use `--force` flag to re-run
- Or delete old files in `data/corners/`
