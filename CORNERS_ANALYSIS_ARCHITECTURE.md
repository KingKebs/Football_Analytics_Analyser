# Visual Architecture: Corners Analysis with --league ALL and --use-parsed-all

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: ARGUMENT PARSING                                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CLI Input:  python3 corners_analysis.py \                                  │
│              --league ALL \                                                 │
│              --use-parsed-all \                                             │
│              --fixtures-date 20251121 \                                     │
│              --top-n 5 \                                                    │
│              --min-team-matches 5                                           │
│                      │                                                       │
│                      ▼                                                       │
│  ┌─────────────────────────────────┐                                        │
│  │ args.league = 'ALL'              │                                        │
│  │ args.use_parsed_all = True       │                                        │
│  │ args.fixtures_date = '20251121'  │                                        │
│  │ args.top_n = 5                   │                                        │
│  │ args.min_team_matches = 5        │                                        │
│  └─────────────────────────────────┘                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: LEAGUE RESOLUTION                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  if args.league.upper() == 'ALL':                                           │
│      leagues = SUPPORTED_LEAGUES  ← 22 league codes                         │
│  else:                                                                       │
│      leagues = ['E0', 'D1', ...]  ← Custom selection                        │
│                                                                              │
│  Result: leagues = [                                                         │
│      'E0', 'E1', 'E2', 'E3',      ← England (4 tiers)                      │
│      'D1', 'D2',                  ← Germany (2 tiers)                       │
│      'SP1', 'SP2',                ← Spain                                    │
│      'I1', 'I2',                  ← Italy                                    │
│      'F1', 'F2',                  ← France                                   │
│      'N1',                         ← Netherlands                             │
│      'P1',                         ← Portugal                                │
│      'SC0', 'SC1', 'SC2', 'SC3',  ← Scotland (4 tiers)                     │
│      'B1', 'G1', 'T1', 'EC'       ← Belgium, Greece, Turkey, Europe         │
│  ]                                                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: PROCESS EACH LEAGUE (Sequential Loop)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  for lg in leagues:           ← 22 iterations                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ ITERATION 1: League E0 (English Premier League)                     │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │                                                                    │    │
│  │ 1. Find CSV: football-data/all-euro-football/E0.csv              │    │
│  │    └→ Load 100+ matches                                           │    │
│  │                                                                    │    │
│  │ 2. Validate corners columns (HC, AC)                             │    │
│  │    └→ ✓ Present and valid                                        │    │
│  │                                                                    │    │
│  │ 3. Clean & Engineer Features                                     │    │
│  │    ├─ Total_Corners = HC + AC                                    │    │
│  │    ├─ Shots_Per_Foul = Total_Shots / Total_Fouls                │    │
│  │    ├─ Shot_Accuracy = Total_Shots_On_Target / Total_Shots       │    │
│  │    ├─ Interaction terms (HS×AS, HTDiff×Shots, etc)              │    │
│  │    └─ Result: 19 engineered features                             │    │
│  │                                                                    │    │
│  │ 4. Calculate Correlations                                        │    │
│  │    ├─ Est_2H_Corners: 0.9888  ✓ Highest                         │    │
│  │    ├─ Est_1H_Corners: 0.9677                                    │    │
│  │    ├─ HC: 0.6299 (home corners                                  │    │
│  │    └─ ... 15 total shown                                         │    │
│  │                                                                    │    │
│  │ 5. Estimate Half-Split                                           │    │
│  │    ├─ Baseline: 40% 1H, 60% 2H                                  │    │
│  │    ├─ Adjust for halftime goals & match intensity               │    │
│  │    └─ Result: Est_1H_Corner_Ratio ≈ 0.35-0.40                  │    │
│  │                                                                    │    │
│  │ 6. Calculate Team Stats ◄─── CRITICAL FOR FIXTURE PREDICTION    │    │
│  │    ├─ Group by HomeTeam:                                        │    │
│  │    │  └─ Man City: Avg 8.0 corners at home                      │    │
│  │    │  └─ Chelsea: Avg 7.6 corners at home                       │    │
│  │    │  └─ Newcastle: Avg 7.2 corners at home                     │    │
│  │    │  └─ ... (20 teams total)                                   │    │
│  │    ├─ Group by AwayTeam:                                        │    │
│  │    │  └─ Arsenal: Avg 8.0 corners away                          │    │
│  │    │  └─ Brentford: Avg 6.8 corners away                        │    │
│  │    │  └─ ... (20 teams total)                                   │    │
│  │    └─ Store: analyzers['E0'].team_stats = (home_stats, away_stats)  │
│  │                                                                    │    │
│  │ 7. Optional: --train-model                                       │    │
│  │    ├─ Linear Regression CV: R²=0.45                             │    │
│  │    ├─ RandomForest CV: R²=0.52 ← Better                         │    │
│  │    └─ XGBoost CV: R²=0.54 (if installed)                        │    │
│  │                                                                    │    │
│  │ 8. Optional: --top-n 5                                           │    │
│  │    ├─ Display Top 5 Home Teams:                                 │    │
│  │    │  1. Man City: 8.0 corners                                  │    │
│  │    │  2. Chelsea: 7.6 corners                                   │    │
│  │    │  3. Newcastle: 7.2 corners                                 │    │
│  │    │  4. Nott'm Forest: 6.8 corners                             │    │
│  │    │  5. Arsenal: 6.6 corners                                   │    │
│  │    └─ Display Top 5 Away Teams                                  │    │
│  │                                                                    │    │
│  │ 9. Store Analyzer in Dict                                        │    │
│  │    └─ analyzers['E0'] = CornersAnalyzer(...)  ✓ SAVED           │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ ITERATION 2: League E1 (Championship) → analyzers['E1'] = ...    │    │
│  │ ITERATION 3: League E2 (League One) → analyzers['E2'] = ...      │    │
│  │ ...                                                                │    │
│  │ ITERATION 22: League EC (European) → analyzers['EC'] = ...       │    │
│  │                                                                    │    │
│  │ Result: analyzers = {                                             │    │
│  │    'E0': CornersAnalyzer(with_team_stats),                        │    │
│  │    'E1': CornersAnalyzer(with_team_stats),                        │    │
│  │    'D1': CornersAnalyzer(with_team_stats),                        │    │
│  │    'SP1': CornersAnalyzer(with_team_stats),                       │    │
│  │    ...22 total...                                                 │    │
│  │ }                                                                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: LOAD PARSED FIXTURES (if --use-parsed-all)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  date_str = '20251121'                                                       │
│           │                                                                  │
│           ▼                                                                  │
│  ┌────────────────────────────────────┐                                    │
│  │ Try: data/todays_fixtures_20251121 │                                    │
│  │  .csv    OR    .json               │                                    │
│  └────────────────┬───────────────────┘                                    │
│                   │                                                        │
│                   ├─ Found: todays_fixtures_20251121.json                 │
│                   │         ✓ Load with pd.read_json()                    │
│                   │                                                        │
│                   ├─ Not found: Try newest: todays_fixtures_20251120.json │
│                   │                                                        │
│                   └─ Normalize columns:                                    │
│                      'home' → 'HomeTeam'                                   │
│                      'away' → 'AwayTeam'                                   │
│                      'league' → 'League'                                   │
│                                                                              │
│  Result: fixtures_df with 25 rows                                           │
│  ┌─────────────────────────────────────────────────────────┐              │
│  │ Date       HomeTeam      AwayTeam      League           │              │
│  ├─────────────────────────────────────────────────────────┤              │
│  │ 2025-11-21 Arsenal       Chelsea       E0               │              │
│  │ 2025-11-21 Man City      Liverpool     E0               │              │
│  │ 2025-11-21 Real Madrid   Barcelona     SP1              │              │
│  │ 2025-11-21 Bayern Munich Dortmund      D1               │              │
│  │ 2025-11-21 Unknown Team  Unknown Team  XY (not loaded)  │              │
│  │ ...        ...           ...           ...              │              │
│  └─────────────────────────────────────────────────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: PREDICT CORNERS FOR EACH FIXTURE                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  for _, row in fixtures_df.iterrows():      ← 25 fixtures                  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ FIXTURE 1: Arsenal vs Chelsea (E0)                                 │    │
│  ├────────────────────────────────────────────────────────────────────┤    │
│  │                                                                    │    │
│  │ Parse:    home='Arsenal', away='Chelsea', league='E0'            │    │
│  │                                                                    │    │
│  │ CHECK 1: Teams present?                                           │    │
│  │ ✓ Arsenal & Chelsea both non-empty                               │    │
│  │                                                                    │    │
│  │ CHECK 2: League in analyzers?                                    │    │
│  │ ✓ 'E0' in analyzers dict                                         │    │
│  │   analyzer = analyzers['E0']                                     │    │
│  │                                                                    │    │
│  │ CHECK 3: Infer league by team presence (if league missing)       │    │
│  │ (Skipped - we have league)                                        │    │
│  │                                                                    │    │
│  │ CHECK 4: Min match history?                                       │    │
│  │ ✓ Arsenal: 20 matches (>= 5)                                     │    │
│  │ ✓ Chelsea: 20 matches (>= 5)                                     │    │
│  │                                                                    │    │
│  │ PREDICT:  analyzer.predict_match_corners('Arsenal', 'Chelsea')   │    │
│  │                                                                    │    │
│  │ Step 1: Get team stats from E0 analyzer:                         │    │
│  │   home_stats['Arsenal'] = {                                      │    │
│  │       'Avg_Corners_For': 6.6,  ← Arsenal avg at home             │    │
│  │       'Avg_Corners_Against': 2.8,  ← Arsenal avg against away    │    │
│  │       'Matches': 20                                              │    │
│  │   }                                                               │    │
│  │   away_stats['Chelsea'] = {                                      │    │
│  │       'Avg_Corners_For': 5.8,  ← Chelsea avg away                │    │
│  │       'Avg_Corners_Against': 3.2,  ← Chelsea avg against home    │    │
│  │       'Matches': 20                                              │    │
│  │   }                                                               │    │
│  │                                                                    │    │
│  │ Step 2: Calculate expected corners:                              │    │
│  │   exp_home = (6.6 + 3.2) / 2 = 4.9 corners                      │    │
│  │   exp_away = (5.8 + 2.8) / 2 = 4.3 corners                      │    │
│  │   total = 4.9 + 4.3 = 9.2 corners                               │    │
│  │                                                                    │    │
│  │ Step 3: Split into 1H/2H:                                        │    │
│  │   ratio = 0.40 (40% first half)                                  │    │
│  │   1H = 9.2 * 0.40 = 3.68 corners                                │    │
│  │   2H = 9.2 * 0.60 = 5.52 corners                                │    │
│  │                                                                    │    │
│  │ RESULT:                                                            │    │
│  │ {                                                                  │    │
│  │   'home_team': 'Arsenal',                                        │    │
│  │   'away_team': 'Chelsea',                                        │    │
│  │   'expected_home_corners': 4.90,                                 │    │
│  │   'expected_away_corners': 4.30,                                 │    │
│  │   'expected_total_corners': 9.20,                                │    │
│  │   'est_1h_ratio': 0.400,                                         │    │
│  │   'expected_1h_corners': 3.68,                                   │    │
│  │   'expected_2h_corners': 5.52,                                   │    │
│  │   'league_code': 'E0',                                           │    │
│  │   'source_file': 'todays_fixtures_20251121'                      │    │
│  │ }                                                                  │    │
│  │ ✓ Added to predictions[]                                          │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ FIXTURE 2: Man City vs Liverpool (E0)                              │    │
│  │ ✓ PASSED all checks → Added to predictions[]                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ FIXTURE 3: Real Madrid vs Barcelona (SP1)                          │    │
│  │ ✓ PASSED all checks → Added to predictions[]                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ FIXTURE 4: Unknown Team 1 vs Unknown Team 2 (XY)                   │    │
│  │ ✗ FAILED CHECK 2: 'XY' not in analyzers                           │    │
│  │ ✗ FAILED CHECK 3: Teams not found in any league                  │    │
│  │ → SKIPPED: {reason: 'no_inference'}                               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ FIXTURE 5: Obscure Team vs Rare Team (J1 - NOT processed)         │    │
│  │ ✗ FAILED CHECK 2: 'J1' not in analyzers (only E0, D1, SP1, ...)  │    │
│  │ → SKIPPED: {league: 'J1', reason: 'league_not_processed'}         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Result Summary:                                                             │
│  ├─ Total fixtures: 25                                                       │
│  ├─ Predictions: 18 ✓                                                        │
│  └─ Skipped: 7                                                               │
│     ├─ league_not_processed: 2                                               │
│     ├─ insufficient_history: 3                                               │
│     ├─ no_inference: 2                                                       │
│     └─ missing_team: 0                                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: SAVE OUTPUT                                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  out_path = 'data/corners/parsed_corners_predictions_20251121.json'        │
│                                                                              │
│  JSON structure:                                                             │
│  {                                                                           │
│    "date": "20251121",                                                      │
│    "predictions": [                                                          │
│      {                                                                       │
│        "home_team": "Arsenal",                                              │
│        "away_team": "Chelsea",                                              │
│        "expected_home_corners": 4.9,                                        │
│        "expected_away_corners": 4.3,                                        │
│        "expected_total_corners": 9.2,                                       │
│        "est_1h_ratio": 0.40,                                                │
│        "expected_1h_corners": 3.68,                                         │
│        "expected_2h_corners": 5.52,                                         │
│        "league_code": "E0",                                                 │
│        "source_file": "todays_fixtures_20251121"                            │
│      },                                                                      │
│      ...17 more predictions...                                              │
│    ],                                                                        │
│    "skipped": [                                                              │
│      {                                                                       │
│        "home": "Team1",                                                      │
│        "away": "Team2",                                                      │
│        "league": "J1",                                                       │
│        "reason": "league_not_processed"                                     │
│      },                                                                      │
│      ...6 more skips...                                                     │
│    ]                                                                         │
│  }                                                                           │
│                                                                              │
│  Output printed:                                                             │
│  ✓ Parsed fixture corner predictions saved: data/corners/parsed_corners_    │
│    predictions_20251121.json                                                │
│  ✓ Predictions: 18 | Skipped: 7                                            │
│  ✓ Skip reasons: league_not_processed=2, insufficient_history=3,           │
│    no_inference=2                                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

### 1. **--league ALL Expansion**
- Single CLI argument expands to 22 league codes
- All leagues analyzed **sequentially** (one after another)
- Each league builds independent team stats
- Analyzers cached for reuse in fixture prediction phase

### 2. **--use-parsed-all Orchestration**
- Waits until **all leagues** fully processed
- Then loads external fixture file (`todays_fixtures_<DATE>.json`)
- Uses cached analyzers to match fixtures to leagues
- Generates predictions using pre-computed team stats

### 3. **Team Stats Reuse**
```
Phase 3: Build once per league
  E0 analyzer: 20 teams × (Avg_Corners_For, Avg_Corners_Against, Matches)
  D1 analyzer: 18 teams × (same stats)
  ...
  
Phase 5: Reuse for predictions
  For fixture "Arsenal vs Chelsea (E0)":
    E0_analyzer.team_stats['Arsenal'][Avg_Corners_For] = 6.6
    E0_analyzer.team_stats['Chelsea'][Avg_Corners_Against] = 3.2
    → Prediction = (6.6 + 3.2) / 2 = 4.9 expected home corners
```

### 4. **Graceful Degradation**
- If league CSV missing → Skip league, move to next
- If fixture league not processed → Skip fixture, log reason
- If team not found → Try to infer from team presence across leagues
- If team insufficient history → Skip, log match count

### 5. **Efficiency**
- All 22 leagues processed **once**
- Team stats computed **once per league**
- Fixture predictions **reuse cached stats** (O(1) lookup)
- Total runtime: ~2-3 minutes for 25 fixtures across 22 leagues


