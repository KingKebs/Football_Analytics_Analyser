---
name: Data Management Skill
description: Handle fixture data, league configs, caching, and data pipeline operations
applies_to: ["src/data_manager.py", "data/**/*.json", "config/**/*.json"]
---

# Data Management Skill

## Data Sources & Locations
- **Fixtures**: `data/todays_fixtures_*.json` (API source, daily)
- **League Configs**: `config/leagues.json` (2-letter codes, settings)
- **League Data**: `data/league_data_*.csv` (historical results)
- **Cache**: `data/cache/` (processed intermediate data)
- **Results**: `data/Results/` (analysis output)
- **Archive**: `data/archive/`, `data/archived/` (old data, backups)

## League Code Mappings
```json
{
  "E0": "Premier League (England)",
  "E1": "Championship (England)",
  "E2": "League One (England)", 
  "E3": "League Two (England)",
  "F1": "Ligue 1 (France)",
  "I1": "Serie A (Italy)",
  "N1": "Eredivisie (Netherlands)",
  "SP1": "La Liga (Spain)"
}
```

## Data Manager (`src/data_manager.py`)
Key functions:
- `load_fixtures(date)` - Get fixtures for specific date
- `load_league_data(league_code)` - Load historical results
- `load_league_config()` - Load league mappings
- `cache_data(key, data)` - Store processed data
- `get_cached_data(key)` - Retrieve cached data

## Fixture Data Structure
```json
{
  "fixtures": [
    {
      "match_id": "...",
      "home": "Team",
      "away": "Team",
      "date": "YYYY-MM-DD HH:MM",
      "league": "E0",
      "odds": {
        "home": 1.5,
        "draw": 3.0,
        "away": 2.5
      }
    }
  ]
}
```

## Pipeline Operations
1. **Fetch**: API → `data/raw/upcomingMatches.json`
2. **Parse**: Filter by date/league
3. **Enrich**: Add odds, market data
4. **Cache**: Store intermediate results
5. **Analyze**: Process for predictions
6. **Output**: Generate results

## Common Tasks
- **Update Fixtures**: `python test_fixtures_download.py`
- **Add League**: Update `config/leagues.json`, fetch data
- **Clean Cache**: Remove old `data/cache/` entries
- **Archive Old Data**: Move to `data/archive/`
- **Validate Data**: Check CSV integrity, JSON format

## Troubleshooting
- **Missing Fixtures**: Check API, verify date format
- **Bad League Code**: Update `config/leagues.json`
- **Cache Stale**: Clear and regenerate
- **Data Corruption**: Compare against backups in archive
