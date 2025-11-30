#!/usr/bin/env python3
"""
Automated Football Fixtures Downloader

Automatically downloads today's football matches from free/open-source APIs
and updates the upcomingMatches.json file.

Free APIs used (with fallbacks):
1. Football-Data.org (free tier: 10 calls/min, 100 calls/day)
2. API-Football (RapidAPI free tier: 100 calls/day)
3. OpenLigaDB (completely free, German leagues)
4. TheSportsDB (free tier available)

Usage:
    python fixtures_downloader.py --update-today
    python fixtures_downloader.py --date 2025-11-30
    python fixtures_downloader.py --leagues Championship,"League One","League Two"
    python fixtures_downloader.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import time

# Optional imports for API clients
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available - API functionality disabled")

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class FixturesDownloader:
    """Download football fixtures from multiple free APIs with fallbacks"""

    def __init__(self):
        self.config = self._load_config()
        self.session = requests.Session() if REQUESTS_AVAILABLE else None

        # Rate limiting
        self.last_api_call = {}
        self.min_api_interval = 1.0  # Min seconds between API calls

    def _load_config(self) -> Dict:
        """Load API configuration from environment or config file"""
        config = {
            'football_data_api_key': os.getenv('FOOTBALL_DATA_API_KEY'),
            'rapid_api_key': os.getenv('RAPID_API_KEY'),
            'output_file': 'data/raw/upcomingMatches.json',
            'backup_file': 'data/raw/upcomingMatches_backup.json',
            'cache_file': 'data/cache/fixtures_cache.json',
            'cache_hours': 2,  # Cache fixtures for 2 hours
        }

        # Create directories if they don't exist
        for file_path in [config['output_file'], config['backup_file'], config['cache_file']]:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        return config

    def _rate_limit(self, api_name: str):
        """Enforce rate limiting between API calls"""
        if api_name in self.last_api_call:
            elapsed = time.time() - self.last_api_call[api_name]
            if elapsed < self.min_api_interval:
                sleep_time = self.min_api_interval - elapsed
                logger.debug(f"Rate limiting {api_name}: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self.last_api_call[api_name] = time.time()

    def _make_api_request(self, url: str, headers: Dict = None, params: Dict = None, api_name: str = "unknown") -> Optional[Dict]:
        """Make rate-limited API request with error handling"""
        if not self.session:
            logger.error("requests library not available")
            return None

        self._rate_limit(api_name)

        try:
            logger.debug(f"API call to {api_name}: {url}")
            response = self.session.get(url, headers=headers or {}, params=params or {}, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"{api_name} API error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"{api_name} JSON decode error: {e}")
            return None

    def _get_football_data_fixtures(self, date: str, leagues: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Get fixtures from Football-Data.org API (free tier: 10 calls/min, 100/day)
        Docs: https://www.football-data.org/documentation/quickstart
        """
        fixtures = {}

        if not self.config['football_data_api_key']:
            logger.info("Football-Data.org API key not found, skipping")
            return fixtures

        headers = {'X-Auth-Token': self.config['football_data_api_key']}

        # Competition IDs for English leagues
        competitions = {
            'Championship': 2016,  # EFL Championship
            'League One': 2017,    # EFL League One
            'League Two': 2018,    # EFL League Two
            'Premier League': 2021  # Premier League
        }

        target_leagues = leagues or competitions.keys()

        for league_name in target_leagues:
            if league_name not in competitions:
                continue

            comp_id = competitions[league_name]
            url = f"https://api.football-data.org/v4/competitions/{comp_id}/matches"
            params = {
                'dateFrom': date,
                'dateTo': date,
                'status': 'SCHEDULED'
            }

            data = self._make_api_request(url, headers, params, f"Football-Data.org-{league_name}")

            if data and 'matches' in data:
                league_fixtures = []
                for match in data['matches']:
                    fixture = {
                        'time': self._extract_time(match.get('utcDate')),
                        'home': self._clean_team_name(match['homeTeam']['name']),
                        'away': self._clean_team_name(match['awayTeam']['name']),
                        'status': match.get('status', 'SCHEDULED'),
                        'source': 'football-data.org'
                    }
                    league_fixtures.append(fixture)

                if league_fixtures:
                    fixtures[league_name] = league_fixtures
                    logger.info(f"Football-Data.org: Found {len(league_fixtures)} fixtures for {league_name}")

        return fixtures

    def _get_api_football_fixtures(self, date: str, leagues: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Get fixtures from API-Football via RapidAPI (free tier: 100 calls/day)
        Docs: https://rapidapi.com/api-sports/api/api-football
        """
        fixtures = {}

        if not self.config['rapid_api_key']:
            logger.info("RapidAPI key not found, skipping API-Football")
            return fixtures

        headers = {
            'X-RapidAPI-Key': self.config['rapid_api_key'],
            'X-RapidAPI-Host': 'api-football-v1.p.rapidapi.com'
        }

        # League IDs for English leagues
        leagues_map = {
            'Championship': 40,    # EFL Championship
            'League One': 41,      # EFL League One
            'League Two': 42,      # EFL League Two
            'Premier League': 39   # Premier League
        }

        target_leagues = leagues or leagues_map.keys()

        for league_name in target_leagues:
            if league_name not in leagues_map:
                continue

            league_id = leagues_map[league_name]
            url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
            params = {
                'date': date,
                'league': league_id,
                'season': 2025  # Current season
            }

            data = self._make_api_request(url, headers, params, f"API-Football-{league_name}")

            if data and 'response' in data:
                league_fixtures = []
                for match in data['response']:
                    fixture = {
                        'time': self._extract_time(match['fixture']['date']),
                        'home': self._clean_team_name(match['teams']['home']['name']),
                        'away': self._clean_team_name(match['teams']['away']['name']),
                        'status': match['fixture'].get('status', {}).get('short', 'NS'),
                        'source': 'api-football'
                    }
                    league_fixtures.append(fixture)

                if league_fixtures:
                    fixtures[league_name] = league_fixtures
                    logger.info(f"API-Football: Found {len(league_fixtures)} fixtures for {league_name}")

        return fixtures

    def _get_thesportsdb_fixtures(self, date: str, leagues: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Get fixtures from TheSportsDB API (free tier available)
        Docs: https://www.thesportsdb.com/api.php
        """
        fixtures = {}

        # TheSportsDB league IDs
        leagues_map = {
            'Championship': '4328',   # English League Championship
            'League One': '4329',     # English League One
            'League Two': '4330',     # English League Two
            'Premier League': '4328'  # English Premier League
        }

        target_leagues = leagues or leagues_map.keys()

        for league_name in target_leagues:
            if league_name not in leagues_map:
                continue

            # TheSportsDB uses different date format
            formatted_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
            url = f"https://www.thesportsdb.com/api/v1/json/3/eventsday.php"
            params = {
                'd': formatted_date,
                'l': leagues_map[league_name]
            }

            data = self._make_api_request(url, params=params, api_name=f"TheSportsDB-{league_name}")

            if data and 'events' in data and data['events']:
                league_fixtures = []
                for event in data['events']:
                    # Skip if not football/soccer
                    if event.get('strSport') != 'Soccer':
                        continue

                    fixture = {
                        'time': self._extract_time_from_thesportsdb(event.get('strTime'), event.get('strDate')),
                        'home': self._clean_team_name(event.get('strHomeTeam', '')),
                        'away': self._clean_team_name(event.get('strAwayTeam', '')),
                        'status': event.get('strStatus', 'Not Started'),
                        'source': 'thesportsdb'
                    }
                    league_fixtures.append(fixture)

                if league_fixtures:
                    fixtures[league_name] = league_fixtures
                    logger.info(f"TheSportsDB: Found {len(league_fixtures)} fixtures for {league_name}")

        return fixtures

    def _extract_time(self, datetime_str: str) -> str:
        """Extract time from various datetime formats"""
        if not datetime_str:
            return "TBD"

        try:
            # Try ISO format first
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime('%H:%M')
        except (ValueError, AttributeError):
            try:
                # Try other common formats
                for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                    dt = datetime.strptime(datetime_str, fmt)
                    return dt.strftime('%H:%M')
            except ValueError:
                logger.debug(f"Could not parse datetime: {datetime_str}")
                return "TBD"

    def _extract_time_from_thesportsdb(self, time_str: str, date_str: str) -> str:
        """Extract time from TheSportsDB format"""
        if time_str and time_str != "":
            try:
                # TheSportsDB time format is usually HH:MM:SS
                time_parts = time_str.split(':')
                return f"{time_parts[0]}:{time_parts[1]}"
            except (IndexError, ValueError):
                pass
        return "TBD"

    def _clean_team_name(self, name: str) -> str:
        """Clean and standardize team names"""
        if not name:
            return ""

        # Common replacements for consistency
        replacements = {
            'AFC ': '',
            'FC ': '',
            ' FC': '',
            ' United': '',
            ' City': '',
            ' Town': '',
            'Milton Keynes Dons': 'MK Dons',
            'Queens Park Rangers': 'QPR',
            'Sheffield Wednesday': 'Sheffield Wed',
            'Sheffield United': 'Sheffield Utd',
            'Tottenham Hotspur': 'Tottenham',
            'Manchester United': 'Man United',
            'Manchester City': 'Man City',
            'Brighton & Hove Albion': 'Brighton',
            'Wolverhampton Wanderers': 'Wolves',
            'Crystal Palace': 'Crystal Palace',
            'West Bromwich Albion': 'West Brom',
            'Nottingham Forest': "Nott'm Forest",
            'Cambridge United': 'Cambridge Utd',
            'Stockport County': 'Stockport County'
        }

        cleaned = name.strip()
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        return cleaned

    def _merge_fixtures(self, *fixture_sources: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Merge fixtures from multiple sources, preferring first source for duplicates"""
        merged = {}

        for fixtures in fixture_sources:
            for league, matches in fixtures.items():
                if league not in merged:
                    merged[league] = []

                # Add matches, avoiding duplicates
                for match in matches:
                    match_key = f"{match['home']}_vs_{match['away']}"
                    existing_keys = [f"{m['home']}_vs_{m['away']}" for m in merged[league]]

                    if match_key not in existing_keys:
                        merged[league].append(match)

        return merged

    def _load_cache(self) -> Optional[Dict]:
        """Load cached fixtures if still valid"""
        try:
            if os.path.exists(self.config['cache_file']):
                with open(self.config['cache_file'], 'r') as f:
                    cache = json.load(f)

                cache_time = datetime.fromisoformat(cache.get('timestamp', ''))
                age_hours = (datetime.now(timezone.utc) - cache_time).total_seconds() / 3600

                if age_hours < self.config['cache_hours']:
                    logger.info(f"Using cached fixtures ({age_hours:.1f}h old)")
                    return cache.get('data')
        except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
            pass

        return None

    def _save_cache(self, data: Dict):
        """Save fixtures to cache"""
        try:
            cache_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': data
            }
            with open(self.config['cache_file'], 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def download_fixtures(self, date: str, leagues: List[str] = None, use_cache: bool = True, dry_run: bool = False) -> Dict[str, Dict]:
        """
        Download fixtures for a specific date

        Args:
            date: Date in YYYY-MM-DD format
            leagues: List of league names to download (None for all)
            use_cache: Whether to use cached data if available
            dry_run: If True, don't make actual API calls

        Returns:
            Dictionary with fixtures organized by country and league
        """
        if dry_run:
            logger.info(f"DRY RUN: Would download fixtures for {date}, leagues: {leagues}")
            return {}

        # Check cache first
        if use_cache:
            cached = self._load_cache()
            if cached:
                return cached

        logger.info(f"Downloading fixtures for {date}...")

        # Try multiple APIs in order of preference
        all_fixtures = {}

        # 1. Football-Data.org (best data quality)
        football_data_fixtures = self._get_football_data_fixtures(date, leagues)

        # 2. API-Football (good fallback)
        api_football_fixtures = self._get_api_football_fixtures(date, leagues)

        # 3. TheSportsDB (free fallback)
        thesportsdb_fixtures = self._get_thesportsdb_fixtures(date, leagues)

        # Merge all sources
        merged_fixtures = self._merge_fixtures(
            football_data_fixtures,
            api_football_fixtures,
            thesportsdb_fixtures
        )

        if merged_fixtures:
            # Organize by country/league structure
            all_fixtures = {
                "ENGLAND": {}
            }

            for league_name, matches in merged_fixtures.items():
                all_fixtures["ENGLAND"][league_name] = {
                    "Standings": {},
                    "Fixtures": matches
                }

            # Cache the results
            self._save_cache(all_fixtures)

            total_matches = sum(len(matches) for matches in merged_fixtures.values())
            logger.info(f"Downloaded {total_matches} total fixtures across {len(merged_fixtures)} leagues")
        else:
            logger.warning("No fixtures downloaded from any source")

        return all_fixtures

    def update_upcoming_matches(self, date: str, leagues: List[str] = None, dry_run: bool = False):
        """
        Download fixtures and update the upcomingMatches.json file

        Args:
            date: Date in YYYY-MM-DD format
            leagues: List of league names to update (None for all)
            dry_run: If True, don't save changes
        """
        # Backup existing file
        if os.path.exists(self.config['output_file']) and not dry_run:
            with open(self.config['output_file'], 'r') as f:
                existing_data = json.load(f)
            with open(self.config['backup_file'], 'w') as f:
                json.dump(existing_data, f, indent=2)
            logger.info(f"Backed up existing file to {self.config['backup_file']}")

        # Download new fixtures
        new_fixtures = self.download_fixtures(date, leagues, dry_run=dry_run)

        if dry_run:
            logger.info("DRY RUN: Would update upcomingMatches.json")
            return

        if new_fixtures:
            # Save updated file
            with open(self.config['output_file'], 'w') as f:
                json.dump(new_fixtures, f, indent=2)

            logger.info(f"Updated {self.config['output_file']} with {date} fixtures")

            # Print summary
            for country, leagues_data in new_fixtures.items():
                logger.info(f"{country}:")
                for league, data in leagues_data.items():
                    fixture_count = len(data.get('Fixtures', []))
                    logger.info(f"  {league}: {fixture_count} matches")
        else:
            logger.warning("No fixtures to update")


def main():
    parser = argparse.ArgumentParser(description='Download football fixtures from free APIs')

    # Date options
    parser.add_argument('--date', type=str, help='Date to download fixtures for (YYYY-MM-DD)')
    parser.add_argument('--update-today', action='store_true', help='Update fixtures for today')
    parser.add_argument('--tomorrow', action='store_true', help='Update fixtures for tomorrow')

    # League options
    parser.add_argument('--leagues', type=str, nargs='*',
                       help='Specific leagues to download (e.g., Championship "League One")')

    # Behavior options
    parser.add_argument('--no-cache', action='store_true', help='Skip cache and force fresh download')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded without making changes')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine target date
    if args.update_today:
        target_date = datetime.now().strftime('%Y-%m-%d')
    elif args.tomorrow:
        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    elif args.date:
        target_date = args.date
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
        logger.info("No date specified, using today")

    # Validate date format
    try:
        datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format: {target_date}. Use YYYY-MM-DD")
        sys.exit(1)

    # Create downloader and update fixtures
    downloader = FixturesDownloader()

    try:
        downloader.update_upcoming_matches(
            date=target_date,
            leagues=args.leagues,
            dry_run=args.dry_run
        )

        if not args.dry_run:
            logger.info(f"✅ Successfully updated fixtures for {target_date}")
            logger.info(f"📁 Output: {downloader.config['output_file']}")

    except KeyboardInterrupt:
        logger.info("Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
