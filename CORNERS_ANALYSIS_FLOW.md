# Corners Analysis Flow: --league ALL --use-parsed-all Explained

## Command Breakdown

```bash
python3 corners_analysis.py --league ALL --use-parsed-all --fixtures-date 20251121 --top-n 5 --min-team-matches 5
```

---

## Step-by-Step Execution Flow

### PHASE 1: Parse Arguments

```python
def parse_args():
    p = argparse.ArgumentParser(...)
    p.add_argument('--league', default='ALL')           # ← Receives "ALL"
    p.add_argument('--use-parsed-all', action='store_true')  # ← Flag set to True
    p.add_argument('--fixtures-date', default=None)     # ← Gets "20251121"
    p.add_argument('--top-n', type=int, default=0)      # ← Gets 5
    p.add_argument('--min-team-matches', type=int, default=5)  # ← Gets 5
    ...
    return p.parse_args()
```

**Result after parsing:**
```python
args.league = 'ALL'
args.use_parsed_all = True
args.fixtures_date = '20251121'
args.top_n = 5
args.min_team_matches = 5
```

---

### PHASE 2: League Resolution in main()

```python
def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    # ✅ KEY DECISION: Check if league is 'ALL'
    if args.league.upper() == 'ALL':
        leagues = SUPPORTED_LEAGUES  # Expand to full list
    else:
        leagues = [l.strip().upper() for l in args.league.split(',')]
```

**SUPPORTED_LEAGUES constant:**
```python
SUPPORTED_LEAGUES = [
    'E0', 'E1', 'E2', 'E3',        # England
    'D1', 'D2',                    # Germany
    'SP1', 'SP2',                  # Spain
    'I1', 'I2',                    # Italy
    'F1', 'F2',                    # France
    'N1',                          # Netherlands
    'P1',                          # Portugal
    'SC0', 'SC1', 'SC2', 'SC3',    # Scotland
    'B1',                          # Belgium
    'G1',                          # Greece
    'T1',                          # Turkey
    'EC'                           # Europe
]
```

**Result after resolution:**
```python
leagues = ['E0', 'E1', 'E2', 'E3', 'D1', 'D2', 'SP1', 'SP2', 'I1', 'I2', 'F1', 'F2', 'N1', 'P1', 'SC0', 'SC1', 'SC2', 'SC3', 'B1', 'G1', 'T1', 'EC']
```

---

### PHASE 3: Process Each League

```python
    summaries = []
    analyzers = {}  # ← Will store analyzers for later fixture matching
    
    for lg in leagues:  # ← Loop through all 22 leagues
        summary = process_league(lg, args)  # Step 3a
        summaries.append(summary)
        
        # ✅ KEY: If --use-parsed-all is enabled AND league processed successfully
        if args.use_parsed_all and summary.get('status') not in ('missing','load_failed','invalid_corners'):
            csv_path = find_league_csv(lg) or args.file
            if csv_path and os.path.isfile(csv_path):
                # Step 3b: Light rebuild for team stats
                analyzer = CornersAnalyzer(csv_path, league_code=lg)
                if analyzer.load_data() is not None and analyzer.validate_corners_data():
                    analyzer.clean_features()
                    analyzer.engineer_features()
                    analyzer.estimate_half_split()
                    analyzer.calculate_team_stats()
                    analyzers[lg] = analyzer  # ← Store for fixture lookup
```

#### Step 3a: process_league(lg, args)

```python
def process_league(league_code: str, args) -> dict:
    csv_path = find_league_csv(league_code) or args.file
    if not csv_path or not os.path.isfile(csv_path):
        return {'league': league_code, 'status': 'missing'}
    
    analyzer = CornersAnalyzer(csv_path, league_code=league_code)
    if analyzer.load_data() is None:
        return {'league': league_code, 'status': 'load_failed'}
    if not analyzer.validate_corners_data():
        return {'league': league_code, 'status': 'invalid_corners'}
    
    # Full analysis pipeline
    analyzer.clean_features()
    analyzer.engineer_features()
    analyzer.estimate_half_split()
    analyzer.calculate_correlations()  # ← Shows corner patterns
    analyzer.calculate_team_stats()     # ← Computes team strength for corners
    
    # Optional: --train-model flag
    metrics = None
    if args.train_model:
        metrics = analyzer.train_models()  # RF/XGB CV metrics
    
    # Optional: --top-n display
    if args.top_n > 0:
        analyzer.display_top_teams(top_n=args.top_n, ...)
    
    # Optional: single match prediction
    if args.home_team and args.away_team:
        try:
            match_pred = analyzer.predict_match_corners(args.home_team, args.away_team)
            print(json.dumps(match_pred, indent=2))
        except Exception as e:
            print(f"Match prediction failed: {e}")
    
    # Optional: --save-enriched
    if args.save_enriched:
        # Save engineering features CSV
        os.makedirs(args.output_dir, exist_ok=True)
        enriched_path = os.path.join(args.output_dir, f'enriched_{league_code}_{timestamp}.csv')
        analyzer.enriched_df.to_csv(enriched_path, index=False)
    
    return {
        'league': league_code,
        'rows': len(analyzer.df),
        'match_prediction': match_pred,
        'model_metrics': metrics,
    }
```

**Per-league output:**
- Prints corner correlations
- Shows team statistics (top 5 teams by avg corners)
- If --top-n 5 specified: lists top 5 home and top 5 away corner teams
- If --train-model: shows RF/Linear/XGB 5-fold CV metrics

#### Step 3b: Store Analyzer for Fixture Lookup

For each successfully processed league, `analyzers` dict now contains:
```python
analyzers = {
    'E0': CornersAnalyzer(E0_data),   # Has team_stats
    'E1': CornersAnalyzer(E1_data),   # Has team_stats
    'SP1': CornersAnalyzer(SP1_data), # Has team_stats
    'D1': CornersAnalyzer(D1_data),   # Has team_stats
    ...
}
```

Each analyzer in `analyzers[lg]` has pre-computed `.team_stats` containing:
- Home team average corners
- Away team average corners
- Matches count per team
- Fouls, shots, etc.

---

### PHASE 4: Parse Fixtures

```python
    # ✅ Only if --use-parsed-all enabled
    if args.use_parsed_all:
        date_str = args.fixtures_date or datetime.now().strftime('%Y%m%d')  # "20251121"
        fixtures_df = _load_parsed_fixtures(date_str)  # Load from data/todays_fixtures_20251121.json
```

#### _load_parsed_fixtures() logic:

```python
def _load_parsed_fixtures(date_str: str = None, data_dir: str = 'data') -> pd.DataFrame:
    # 1️⃣ Try exact date
    csv_path = os.path.join(data_dir, f'todays_fixtures_{date_str}.csv')      # data/todays_fixtures_20251121.csv
    json_path = os.path.join(data_dir, f'todays_fixtures_{date_str}.json')    # data/todays_fixtures_20251121.json
    
    # 2️⃣ If not found, fallback to newest
    if not os.path.exists(csv_path) and not os.path.exists(json_path):
        candidates = sorted(
            glob(f'{data_dir}/todays_fixtures_*.json') + glob(f'{data_dir}/todays_fixtures_*.csv'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if candidates:
            path = str(candidates[0])  # Most recent file
    
    # 3️⃣ Load (CSV or JSON)
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path)
    
    # 4️⃣ Normalize column names
    # Remap: 'home' → 'HomeTeam', 'away' → 'AwayTeam', 'league' → 'League', etc.
    
    return df
```

**Expected fixture file structure** (data/todays_fixtures_20251121.json):
```json
[
  {
    "Date": "2025-11-21",
    "HomeTeam": "Arsenal",
    "AwayTeam": "Chelsea",
    "League": "E0",
    "Competition": "Premier League"
  },
  {
    "Date": "2025-11-21",
    "HomeTeam": "Real Madrid",
    "AwayTeam": "Barcelona",
    "League": "SP1",
    "Competition": "La Liga"
  },
  ...
]
```

---

### PHASE 5: Predict Corners for Each Fixture

```python
        if fixtures_df.empty:
            print(f"No parsed fixtures found for date {date_str}")
        else:
            print(f"\nParsed fixtures loaded ({len(fixtures_df)}) for date {date_str}")
            predictions = []
            skips = []
            
            # Loop through each fixture row
            for _, row in fixtures_df.iterrows():
                home = row.get('HomeTeam')
                away = row.get('AwayTeam')
                league = str(row.get('League') or '').upper()  # e.g. "E0"
                
                # ✅ CHECK 1: Teams present?
                if not home or not away:
                    skips.append({'reason': 'missing_team'})
                    continue
                
                # ✅ CHECK 2: League in analyzers?
                if league and league not in analyzers:
                    # e.g. parsed fixture says "SP1" but analyzers only has E0, D1, etc
                    skips.append({'league': league, 'reason': 'league_not_processed'})
                    continue
                
                # Get analyzer for this league
                analyzer = analyzers.get(league)
                
                # ✅ CHECK 3: If league not set, try to infer by team presence
                if not analyzer:
                    # Search all leagues for one that has BOTH teams
                    for lg, an in analyzers.items():
                        hs, as_ = an.team_stats  # home_stats, away_stats
                        if home in hs.index and away in as_.index:
                            analyzer = an
                            league = lg
                            break
                
                if not analyzer:
                    skips.append({'reason': 'no_inference'})
                    continue
                
                # ✅ CHECK 4: Minimum match history?
                hs, as_ = analyzer.team_stats
                h_matches = int(hs.loc[home]['Matches']) if home in hs.index else 0
                a_matches = int(as_.loc[away]['Matches']) if away in as_.index else 0
                
                if h_matches < args.min_team_matches or a_matches < args.min_team_matches:
                    skips.append({
                        'reason': 'insufficient_history',
                        'home_matches': h_matches,
                        'away_matches': a_matches
                    })
                    continue
                
                # ✅ ALL CHECKS PASSED: Generate prediction
                try:
                    pred = analyzer.predict_match_corners(home, away)
                    pred['league_code'] = league
                    pred['source_file'] = f'todays_fixtures_{date_str}'
                    predictions.append(pred)
                except Exception as e:
                    skips.append({'reason': 'prediction_error', 'error': str(e)})
```

#### predict_match_corners() Method:

```python
def predict_match_corners(self, home_team: str, away_team: str):
    # Extract team stats computed in Phase 3b
    home_stats, away_stats = self.team_stats
    
    h_row = home_stats.loc[home_team]
    a_row = away_stats.loc[away_team]
    
    # Expected corners = average(team's average for + opponent's average against)
    home_for = float(h_row['Avg_Corners_For'])          # e.g., 8.5 corners when Arsenal at home
    away_conc_by_home = float(a_row['Avg_Corners_Against'])  # e.g., 6.2 corners Arsenal allows away
    
    away_for = float(a_row['Avg_Corners_For'])          # e.g., 7.1 corners when Chelsea away
    home_conc_by_away = float(h_row['Avg_Corners_Against']) # e.g., 5.8 corners Chelsea allows at home
    
    exp_home = (home_for + home_conc_by_away) / 2.0
    exp_away = (away_for + away_conc_by_home) / 2.0
    total = exp_home + exp_away
    
    # 1H/2H split (using median from historical data)
    ratio = float(self.enriched_df['Est_1H_Corner_Ratio'].median())  # e.g., 0.40
    one_h = total * ratio
    two_h = total - one_h
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'expected_home_corners': round(exp_home, 2),      # e.g., 7.2
        'expected_away_corners': round(exp_away, 2),      # e.g., 6.5
        'expected_total_corners': round(total, 2),        # e.g., 13.7
        'est_1h_ratio': round(ratio, 3),                  # e.g., 0.400
        'expected_1h_corners': round(one_h, 2),           # e.g., 5.5
        'expected_2h_corners': round(two_h, 2),           # e.g., 8.2
    }
```

---

### PHASE 6: Save Results

```python
            # Save predictions
            out_dir = args.output_dir or 'data/corners'
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'parsed_corners_predictions_{date_str}.json')
            
            with open(out_path, 'w') as f:
                json.dump({
                    'date': date_str,
                    'predictions': predictions,
                    'skipped': skips
                }, f, indent=2)
            
            print(f"\nParsed fixture corner predictions saved: {out_path}")
            print(f"Predictions: {len(predictions)} | Skipped: {len(skips)}")
            
            if skips:
                reason_counts = {}
                for s in skips:
                    r = s.get('reason', 'unknown')
                    reason_counts[r] = reason_counts.get(r, 0) + 1
                print("Skip reasons: " + ', '.join(f"{k}={v}" for k,v in reason_counts.items()))
    
    return 0
```

**Output file** (data/corners/parsed_corners_predictions_20251121.json):
```json
{
  "date": "20251121",
  "predictions": [
    {
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "expected_home_corners": 7.2,
      "expected_away_corners": 6.5,
      "expected_total_corners": 13.7,
      "est_1h_ratio": 0.400,
      "expected_1h_corners": 5.5,
      "expected_2h_corners": 8.2,
      "league_code": "E0",
      "source_file": "todays_fixtures_20251121"
    },
    ...
  ],
  "skipped": [
    {
      "home": "Team1",
      "away": "Team2",
      "league": "SP1",
      "reason": "league_not_processed"
    },
    ...
  ]
}
```

---

## Key Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│  python3 corners_analysis.py --league ALL --use-parsed-all  │
│           --fixtures-date 20251121 --top-n 5                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Parse Arguments │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ league='ALL'? ✓ │
                    └────────┬────────┘
                             │
                    ┌────────▼──────────────────┐
                    │ Expand to SUPPORTED_LEAGUES│
                    │ [E0, E1, E2, E3, D1, ...] │
                    └────────┬──────────────────┘
                             │
             ┌───────────────┼───────────────┐
             │ For each league lg in list    │
             │ (Loop through 22+ leagues)    │
             └───────────────┼───────────────┘
                             │
                    ┌────────▼──────────────────────────┐
                    │ process_league(lg, args)          │
                    │ • Load CSV for league             │
                    │ • Engineer features               │
                    │ • Calculate team stats            │
                    │ • Display --top-n teams           │
                    └────────┬───────────────────────────┘
                             │
                    ┌────────▼──────────────────┐
                    │ use_parsed_all=True? ✓    │
                    └────────┬──────────────────┘
                             │
                    ┌────────▼──────────────────┐
                    │ Store analyzer in dict    │
                    │ analyzers[lg] = CornersA. │
                    └────────┬──────────────────┘
                             │
          ┌──────────────────┘
          │
    (After all leagues processed)
          │
          ▼
    ┌──────────────────────────┐
    │ Load parsed fixtures     │
    │ data/todays_fixtures_    │
    │ 20251121.json            │
    └────────┬─────────────────┘
             │
    ┌────────▼────────────────────────────┐
    │ For each fixture in parsed_df:      │
    │ ┌─────────────────────────────────┐ │
    │ │ 1. Extract: home, away, league  │ │
    │ │ 2. Find analyzer for league     │ │
    │ │ 3. Check min team matches       │ │
    │ │ 4. Call predict_match_corners() │ │
    │ │ 5. Append to predictions[]      │ │
    │ └─────────────────────────────────┘ │
    └────────┬─────────────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Save to JSON file            │
    │ data/corners/parsed_corners_ │
    │ predictions_20251121.json    │
    └──────────────────────────────┘
```

---

## Example Output

### Terminal Output:
```
[11:47:48] INFO: Task: corners | Leagues: E0,D1,SP1,...
✓ Loaded 100 matches from football-data/all-euro-football/E0.csv (E0)
✓ Corners data validated
✓ 19 new features engineered

CORNER CORRELATIONS (Top 15)
Est_2H_Corners: 0.9888
Est_1H_Corners: 0.9677
HC: 0.6299
...

TEAM STATISTICS (Top 5 by Avg Corners)
Home Teams:
           Avg_Corners_For  Std_Corners_For  Matches
Man City   8.0              3.67             5
Chelsea    7.6              2.41             5
Newcastle  7.2              2.17             5
...

Top 5 Home Teams (Avg Corners For)
Man City: 8.0, Chelsea: 7.6, Newcastle: 7.2, ...

✓ Loaded 100 matches from football-data/all-euro-football/D1.csv (D1)
...

Parsed fixtures loaded (25) for date 20251121

Parsed fixture corner predictions saved: data/corners/parsed_corners_predictions_20251121.json
Predictions: 18 | Skipped: 7
Skip reasons: league_not_processed=2, insufficient_history=5
```

### JSON Output File:
```json
{
  "date": "20251121",
  "predictions": [
    {
      "home_team": "Arsenal",
      "away_team": "Chelsea",
      "expected_home_corners": 7.15,
      "expected_away_corners": 6.45,
      "expected_total_corners": 13.6,
      "est_1h_ratio": 0.400,
      "expected_1h_corners": 5.44,
      "expected_2h_corners": 8.16,
      "league_code": "E0",
      "source_file": "todays_fixtures_20251121"
    },
    ...
  ],
  "skipped": [
    {
      "home": "Team1",
      "away": "Team2",
      "league": "UNKNOWN_LEAGUE",
      "reason": "league_not_processed"
    }
  ]
}
```

---

## Why Fixtures Get Skipped

| Skip Reason | Meaning | Example |
|---|---|---|
| `missing_team` | HomeTeam or AwayTeam is empty | Parsed fixture row incomplete |
| `league_not_processed` | League code in fixture doesn't match any processed league | Fixture says `league="J1"` but only E0,D1 processed |
| `league_not_in_analyzers` | Analyzer dict doesn't have that league | Fixture from SP2 but we only loaded SP1 |
| `no_inference` | Can't find team in any analyzer | Team name doesn't exist in any loaded league |
| `insufficient_history` | Team has <5 matches (default) | Arsenal has 3 matches in dataset |
| `prediction_error` | Corner prediction threw exception | Team lookup failed mid-calculation |

---

## Summary: How It Works

1. **--league ALL** → Expands to all 22 supported leagues (E0, E1, ..., T1, EC)
2. **For each league** → Loads CSV, calculates team corner stats, optionally shows top teams
3. **--use-parsed-all** → After all leagues processed, load `todays_fixtures_20251121.json`
4. **For each fixture** → 
   - Identify league from fixture row
   - Find corresponding league's analyzer (has team stats)
   - Validate teams have sufficient history
   - Predict corners using formula: `(team_avg_for + opponent_avg_against) / 2`
5. **Output** → Save predictions to JSON with skip reasons

The beauty: **All 22 leagues analyzed once, then their stats reused to predict corners across ANY fixtures regardless of league!**


