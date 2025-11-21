# Quick Explanation: --league ALL and --use-parsed-all

## The Command
```bash
python3 corners_analysis.py --league ALL --use-parsed-all --fixtures-date 20251121 --top-n 5 --min-team-matches 5
```

---

## What Each Flag Does

### `--league ALL`
**Action:** Expands to 22 supported leagues
```
ALL → [E0, E1, E2, E3, D1, D2, SP1, SP2, I1, I2, F1, F2, N1, P1, SC0, SC1, SC2, SC3, B1, G1, T1, EC]
```

**Result:** For EACH league:
- Load historical match CSV
- Calculate team corner statistics
- Show top teams by corners
- Store analyzer in memory

**Example output per league:**
```
✓ Loaded 100 matches from E0.csv
CORNER CORRELATIONS (Top 5):
  Est_2H_Corners: 0.9888
  Est_1H_Corners: 0.9677
  HC: 0.6299
  ...
TEAM STATISTICS (Top 5):
  Man City: 8.0 avg corners (home)
  Chelsea: 7.6 avg corners (home)
  ...
```

---

### `--use-parsed-all`
**Action:** After all leagues processed, predict corners for upcoming fixtures

**Workflow:**
1. Load fixture file: `data/todays_fixtures_20251121.json` (25 fixtures)
2. For each fixture, identify home team, away team, league
3. Find corresponding league's analyzer (from Phase 1)
4. Use team stats to predict total corners

**Example prediction:**
```
Fixture: Arsenal (E0) vs Chelsea (E0)
  Arsenal avg corners at home: 6.6
  Chelsea avg corners allowed: 3.2
  Expected: (6.6 + 3.2) / 2 = 4.9 corners
  
  Chelsea avg corners away: 5.8
  Arsenal avg corners allowed: 2.8
  Expected: (5.8 + 2.8) / 2 = 4.3 corners
  
  TOTAL: 4.9 + 4.3 = 9.2 corners
  1H/2H: 3.68 (1H) + 5.52 (2H)
```

---

## The Three Phases

### Phase 1: Process All Leagues (22 loops)
```
For lg in ['E0', 'E1', ..., 'T1', 'EC']:
  - Load CSV for league
  - Engineer 19 features
  - Calculate team stats (per team: avg corners for/against)
  - Store in analyzers[lg]
```

### Phase 2: Load Fixture File
```
Load: data/todays_fixtures_20251121.json
Extract: home_team, away_team, league (from each row)
```

### Phase 3: Predict Corners
```
For each fixture:
  1. Find analyzer for league
  2. Check teams have >= 5 matches (--min-team-matches)
  3. Get team stats: home_avg_for, away_avg_against
  4. Calculate: (home_avg_for + away_avg_against) / 2
  5. Save prediction JSON
```

---

## Why Fixtures Get Skipped

| Reason | Example |
|--------|---------|
| `league_not_processed` | Fixture says league="J1" but we only loaded 22 European leagues |
| `insufficient_history` | Team has only 2 matches; threshold is 5 (--min-team-matches) |
| `no_inference` | Team name doesn't exist in any loaded league's data |
| `missing_team` | HomeTeam or AwayTeam is empty in fixture row |

---

## Output File

**Location:** `data/corners/parsed_corners_predictions_20251121.json`

**Structure:**
```json
{
  "date": "20251121",
  "predictions": [
    {
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "expected_home_corners": 4.90,
      "expected_away_corners": 4.30,
      "expected_total_corners": 9.20,
      "est_1h_ratio": 0.400,
      "expected_1h_corners": 3.68,
      "expected_2h_corners": 5.52,
      "league_code": "E0",
      "source_file": "todays_fixtures_20251121"
    }
  ],
  "skipped": [
    {
      "league": "J1",
      "reason": "league_not_processed"
    }
  ]
}
```

**Summary printed:**
```
Parsed fixtures loaded (25) for date 20251121
Parsed fixture corner predictions saved: data/corners/parsed_corners_predictions_20251121.json
Predictions: 18 | Skipped: 7
Skip reasons: league_not_processed=2, insufficient_history=5
```

---

## Key Insights

**Why use `--league ALL` instead of specific leagues?**
- Maximizes fixture matching (any league fixture can be analyzed)
- Single command covers all supported leagues
- Reuse same team stats across multiple fixtures

**Why use `--use-parsed-all` instead of manual match-by-match?**
- Batch process 25 fixtures at once
- Automatic league inference from fixture file
- Single JSON output with all predictions + skip reasons

**Why does it take 2-3 minutes?**
- 22 league CSVs loaded & analyzed
- 19 features engineered per league
- Team stats computed for ~400 teams
- Fixtures predicted using cached stats (fast)

---

## Files Generated

- **`data/corners/parsed_corners_predictions_20251121.json`** ← Main output
- **`data/corners/model_metrics_E0_<ts>.json`** (if --train-model)
- **`data/corners/enriched_E0_<ts>.csv`** (if --save-enriched)

---

## See Also
- **CORNERS_ANALYSIS_FLOW.md** - Detailed step-by-step execution
- **CORNERS_ANALYSIS_ARCHITECTURE.md** - Visual data flow diagrams

