# Automated Fixtures Download Setup Guide

## Overview

The Football Analytics system now includes automated fixture downloading from multiple free APIs and web scraping sources. This eliminates the need to manually copy data from sites like Flashscore.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install web scraping dependencies
pip install requests beautifulsoup4 lxml

# Or install all requirements
pip install -r requirements.txt
```

### 2. Download Today's Fixtures
```bash
# Download today's fixtures and update upcomingMatches.json
python cli.py --task download-fixtures --update-today

# Or use the direct script
python src/fixtures_downloader.py --update-today
```

### 3. Run Analysis with Fresh Fixtures
```bash
# Download fixtures and run full analysis
python cli.py --task download-fixtures --update-today
python cli.py --task full-league --leagues E1,E2,E3 --enable-double-chance --ml-mode predict
```

## 📡 Data Sources

### Free APIs (No Registration Required)
1. **TheSportsDB** - Completely free, no API key needed
   - Unlimited calls
   - Good coverage of English leagues

### Free APIs (Registration Required)
2. **Football-Data.org** - Free tier: 10 calls/min, 100 calls/day
   - Register at: https://www.football-data.org/client/register
   - High quality data
   - Official league data

3. **API-Football (RapidAPI)** - Free tier: 100 calls/day
   - Register at: https://rapidapi.com/api-sports/api/api-football
   - Comprehensive coverage
   - Real-time updates

### Web Scraping Fallback
4. **BBC Sport** - Web scraping (no API key needed)
   - Used when APIs are unavailable
   - Respectful scraping with rate limiting

## 🔧 Configuration

### Option 1: Environment Variables
```bash
export FOOTBALL_DATA_API_KEY="your_key_here"
export RAPID_API_KEY="your_rapidapi_key_here"
```

### Option 2: .env File
Create `.env` file in project root:
```
FOOTBALL_DATA_API_KEY=your_key_here
RAPID_API_KEY=your_rapidapi_key_here
```

### Option 3: No API Keys (Still Works!)
The system works with free sources even without API keys:
- Uses TheSportsDB (completely free)
- Falls back to web scraping
- Caches results to minimize requests

## 📋 Usage Examples

### Basic Usage
```bash
# Download today's fixtures
python cli.py --task download-fixtures --update-today

# Download specific date
python cli.py --task download-fixtures --date 2025-12-01

# Download tomorrow's fixtures
python cli.py --task download-fixtures --tomorrow

# Specific leagues only
python cli.py --task download-fixtures --leagues "Championship,League One" --update-today

# Force fresh download (skip cache)
python cli.py --task download-fixtures --update-today --no-cache

# Dry run (see what would be downloaded)
python cli.py --task download-fixtures --update-today --dry-run
```

### Automated Daily Workflow
```bash
# Download fixtures and run full analysis
python fixtures_scheduler.py --update-and-analyze

# Daily fixtures update only
python fixtures_scheduler.py --daily-update

# Setup as cron job (runs at 8 AM daily)
0 8 * * * cd /path/to/Football_Analytics_Analyser && python fixtures_scheduler.py --daily-update
```

### Integration with Analysis
```bash
# Complete daily workflow
python cli.py --task download-fixtures --update-today
python cli.py --task full-league --leagues E1,E2,E3 --enable-double-chance --ml-mode predict --parallel-workers 3

# Use fixtures in corner analysis
python cli.py --task download-fixtures --update-today
python cli.py --task corners --use-parsed-all --corners-use-ml-prediction
```

## 📂 Output Structure

The downloader updates `data/raw/upcomingMatches.json` with this structure:

```json
{
  "ENGLAND": {
    "Championship": {
      "Standings": {},
      "Fixtures": [
        {
          "time": "14:30",
          "home": "Leicester",
          "away": "Sheffield Utd",
          "status": "SCHEDULED",
          "source": "football-data.org"
        }
      ]
    },
    "League One": {
      "Standings": {},
      "Fixtures": [...]
    }
  }
}
```

## 🎯 Features

### Multi-Source Reliability
- **Primary**: Uses high-quality APIs when available
- **Fallback**: Automatic fallback to web scraping
- **Caching**: Reduces API calls with intelligent caching
- **Merge Logic**: Combines data from multiple sources

### Rate Limiting & Respect
- **API Rate Limiting**: Respects API limits automatically
- **Web Scraping Ethics**: 2-second delays between requests
- **Cache System**: 2-hour cache to minimize requests
- **Error Handling**: Graceful degradation when sources fail

### Team Name Standardization
- **Consistent Names**: Automatically standardizes team names
- **Common Replacements**: Handles "AFC", "FC", "United" variations
- **League Compatibility**: Names match existing analysis data

## 🔄 Automation Options

### Option 1: Cron Job (Linux/Mac)
```bash
# Edit crontab
crontab -e

# Add daily 8 AM update
0 8 * * * cd /path/to/Football_Analytics_Analyser && python fixtures_scheduler.py --daily-update

# Add with analysis at 9 AM
0 9 * * * cd /path/to/Football_Analytics_Analyser && python fixtures_scheduler.py --update-and-analyze
```

### Option 2: Windows Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Daily at 8:00 AM
4. Action: Start Program
5. Program: `python`
6. Arguments: `fixtures_scheduler.py --daily-update`
7. Start in: `/path/to/Football_Analytics_Analyser`

### Option 3: Manual Integration
```python
# In your own automation script
from src.fixtures_downloader import FixturesDownloader

downloader = FixturesDownloader()
downloader.update_upcoming_matches("2025-11-30")
```

## 🐛 Troubleshooting

### API Issues
```bash
# Check if APIs are working
python src/fixtures_downloader.py --date 2025-11-30 --verbose

# Force web scraping fallback
python src/fixtures_downloader.py --date 2025-11-30 --verbose
# (Remove API keys from environment)
```

### Dependencies Issues
```bash
# Install missing dependencies
pip install requests beautifulsoup4 lxml

# Check if web scraping works
python -c "import requests, bs4; print('Web scraping ready')"
```

### Cache Issues
```bash
# Clear cache and force fresh download
rm data/cache/fixtures_cache.json
python cli.py --task download-fixtures --update-today --no-cache
```

### File Permission Issues
```bash
# Ensure directories exist and are writable
mkdir -p data/cache data/raw
chmod 755 data/cache data/raw
```

## 📈 Performance & Costs

### API Usage (with free tiers)
- **Football-Data.org**: 100 calls/day free
- **API-Football**: 100 calls/day free  
- **TheSportsDB**: Unlimited free
- **Web Scraping**: Unlimited (respectful)

### Typical Daily Usage
- **3 leagues × 1 call each = 3 API calls**
- **Well within all free tiers**
- **Cache reduces repeat calls**
- **Fallbacks prevent failures**

### Optimization
- **Cache**: 2-hour cache reduces API calls
- **Smart Merging**: Combines multiple sources
- **Rate Limiting**: Prevents API throttling
- **Parallel Processing**: Fast analysis after download

## 🔮 Future Enhancements

### Planned Features
- **More Leagues**: Add support for other countries
- **Live Scores**: Real-time score updates
- **Odds Integration**: Automatic odds scraping
- **Notifications**: Slack/email alerts
- **Database**: Store historical fixture data

### API Expansion
- **Additional Sources**: More free APIs
- **Odds APIs**: Free odds comparison
- **Statistics APIs**: Team form data
- **Weather APIs**: Match conditions

## 📞 Support

### Getting Help
1. Check logs in `logs/fixtures_scheduler_*.log`
2. Run with `--verbose` for debug info
3. Test individual components with `--dry-run`
4. Check API status at provider websites

### Common Solutions
- **No fixtures**: Check date format (YYYY-MM-DD)
- **API errors**: Verify API keys and quotas
- **Web scraping fails**: Check internet connection
- **File errors**: Ensure directory permissions
