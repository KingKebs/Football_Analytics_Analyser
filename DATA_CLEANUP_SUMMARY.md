# Data Directory Cleanup - Summary

## What Was Done

Ran `organize_structure.py` to clean up the cluttered `data/` directory.

## Before (Cluttered)
```
data/
  ├── 80+ mixed CSV/JSON files (corners, league analysis, fixtures all together)
  ├── Old suggestion files from October
  ├── Duplicate corner analysis CSVs
  └── No clear organization
```

## After (Organized)
```
data/
  ├── corners/           (89 files - all corner analysis & predictions)
  ├── league_analysis/   (25 files - full league suggestions)
  ├── fixtures/          (7 files - parsed match fixtures)
  ├── archived/          (11 files - old files >7 days automatically archived)
  └── *.csv              (21 files - league data tables kept in root)
```

## How to Use

### Clean Up Anytime
```bash
# Preview changes first
python3 organize_structure.py --dry-run

# Apply organization
python3 organize_structure.py
```

### What Gets Organized

1. **Corners Files** → `data/corners/`
   - `corners_analysis_*.csv`
   - `corners_correlations_*.json`
   - `team_stats_*.json`
   - `match_prediction_*.json`
   - `batch_predictions_*.json`

2. **League Analysis** → `data/league_analysis/`
   - `full_league_suggestions_*.json` (recent, <7 days)

3. **Fixtures** → `data/fixtures/`
   - `todays_fixtures_*.*` (recent, <7 days)

4. **Archived** → `data/archived/`
   - Old league suggestions (>7 days)
   - Old fixtures (>7 days)
   - Old individual match suggestions

5. **Root** (unchanged)
   - `league_data_*.csv` (always kept in data/)
   - Other analysis CSVs

## Benefits

✅ **Easy to find files** - Clear subdirectories by purpose  
✅ **Automatic archiving** - Files >7 days moved to archived/  
✅ **No data loss** - All files preserved, just reorganized  
✅ **Safe to re-run** - Idempotent (won't break if run multiple times)  
✅ **Dry-run mode** - Preview changes before applying  

## Integration with Workflows

All scripts automatically use the new structure:
- `corners_analysis.py` → writes to `data/corners/`
- `automate_corner_predictions.py` → reads from/writes to `data/corners/`
- `parse_match_log.py` → writes to `data/` (then organized to fixtures/)
- `automate_football_analytics_fullLeague.py` → writes to `data/` (then organized to league_analysis/)

## Maintenance

Run this periodically (weekly recommended) to keep data organized:
```bash
python3 organize_structure.py
```

Or add to your workflow:
```bash
# At the start of your analysis session
python3 organize_structure.py --dry-run  # check what would change
python3 organize_structure.py            # apply if looks good
```

