# File Management System - Corner Predictions

## Problem Solved

**Before:** Multiple timestamped files created every run on the same day:
```
batch_predictions_E2+E3_20251115_001326.json
batch_predictions_E2+E3_20251115_003845.json  
batch_predictions_E2+E3_20251115_010522.json  ← Clutter!
corners_analysis_E2_20251115_001330.csv
corners_analysis_E2_20251115_003850.csv       ← Duplicates!
```

**After:** Single latest file per day + automatic cleanup:
```
batch_predictions_E2+E3_20251115_010522.json  ← Latest kept
corners_analysis_E2_20251115_010530.csv       ← Latest kept
```

## New Features

### 1. Automatic Duplicate Removal
When creating a new prediction file for the day, **automatically removes older files from the same day** with the same pattern.

**How it works:**
- Creates new file with timestamp (e.g., `batch_predictions_E2_20251115_143022.json`)
- Finds all other files matching `batch_predictions_E2_20251115_*.json`
- Keeps only the newest one
- Removes the rest

**Output:**
```
✓ Exported 5 predictions to: data/corners/batch_predictions_E2_20251115_143022.json
🗑️  Removed duplicate: batch_predictions_E2_20251115_140510.json
🗑️  Removed duplicate: batch_predictions_E2_20251115_141205.json
```

### 2. Reuse Today's Data (--reuse-today)
Skip regenerating predictions if they already exist for today.

**Usage:**
```bash
python3 automate_corner_predictions.py --input <log> --leagues E2,E3 --auto --reuse-today
```

**Behavior:**
- Checks if `batch_predictions_E2+E3_20251115_*.json` exists
- If yes: reuses existing file, skips prediction generation
- If no: generates fresh predictions

**Output when reusing:**
```
======================================================================
✓ Reusing existing predictions from: batch_predictions_E2+E3_20251115_140510.json
======================================================================
```

### 3. Automatic Old File Cleanup
At the end of every workflow run, **removes files older than 7 days**.

**Files cleaned:**
- `batch_predictions_*.json`
- `match_prediction_*.json`
- `corners_analysis_*.csv`
- `corners_correlations_*.json`
- `team_stats_*.json`

**Output:**
```
🗑️  Cleaned up 47 old file(s) older than 7 days
```

## Usage Examples

### Basic (default behavior - auto-cleanup duplicates)
```bash
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --leagues E2,E3 --auto
```
- Generates predictions
- Removes duplicate files from today
- Cleans up files >7 days old

### Reuse existing predictions from today
```bash
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --leagues E2,E3 --auto --reuse-today
```
- Checks for existing predictions
- If found, reuses them instead of regenerating
- Useful for quick re-runs or testing

### Force fresh analysis + predictions
```bash
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --leagues E2,E3 --auto --force
```
- Ignores existing analysis from today
- Re-runs corners_analysis.py
- Generates fresh predictions
- Still removes duplicates and cleans up old files

## File Naming Convention

All files now follow this pattern:
```
{prefix}_{league}_{YYYYMMDD}_{HHMMSS}.{ext}
```

Examples:
- `batch_predictions_E2_20251115_143022.json`
- `batch_predictions_E2+E3_20251115_143025.json`
- `corners_analysis_E2_20251115_143020.csv`
- `team_stats_E3_20251115_143030.json`

**Why this format?**
- Date (YYYYMMDD) enables day-based duplicate detection
- Time (HHMMSS) ensures uniqueness within the same minute
- Sortable chronologically by filename

## Benefits

✅ **No more directory flooding** - Only 1 file per pattern per day  
✅ **Automatic cleanup** - Files >7 days removed automatically  
✅ **Reusable predictions** - Skip regeneration with `--reuse-today`  
✅ **Fast re-runs** - Don't reprocess if data exists  
✅ **Chronological history** - Easy to find today's vs yesterday's predictions  

## Technical Details

### Duplicate Removal Logic
```python
def remove_duplicate_daily_files(output_dir: str, pattern_prefix: str):
    today = datetime.now().strftime('%Y%m%d')
    pattern = os.path.join(output_dir, f"{pattern_prefix}*{today}*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    
    if len(files) > 1:
        # Keep the latest, remove the rest
        for f in files[:-1]:
            os.remove(f)
```

Called automatically after each export.

### Old File Cleanup Logic
```python
def cleanup_old_daily_files(output_dir: str, keep_days: int = 7):
    cutoff = datetime.now() - timedelta(days=keep_days)
    cutoff_ts = cutoff.timestamp()
    
    for pattern in ['batch_predictions_*.json', 'match_prediction_*.json', ...]:
        for fpath in glob.glob(os.path.join(output_dir, pattern)):
            if os.path.getmtime(fpath) < cutoff_ts:
                os.remove(fpath)
```

Called once at the end of `main()`.

### Reuse Today Logic
```python
def export_batch_predictions(..., reuse_today: bool = False):
    today = datetime.now().strftime('%Y%m%d')
    today_pattern = f"batch_predictions_{safe_label}_{today}_*.json"
    existing = find_today_file(os.path.join(output_dir, today_pattern))
    
    if existing and reuse_today:
        return existing  # Skip regeneration
    
    # Generate new file...
```

## Customization

Want to change retention period? Edit the call in `main()`:
```python
# Keep files for 14 days instead of 7
cleanup_old_daily_files('data/corners', keep_days=14)
```

Want to disable auto-cleanup? Comment out the line:
```python
# cleanup_old_daily_files('data/corners', keep_days=7)
```

## Summary

The new system ensures:
1. **One latest file per day** (duplicates auto-removed)
2. **Optional reuse** with `--reuse-today`
3. **Automatic retention management** (7-day default)
4. **Clean, organized directories**

No manual file management needed!

