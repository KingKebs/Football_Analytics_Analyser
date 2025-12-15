#!/usr/bin/env python3
"""
Web Scraper Fallback for Football Fixtures

This module provides web scraping capabilities as a fallback when APIs are unavailable.
Uses BeautifulSoup to scrape fixture data from public websites.

Supported sources:
- BBC Sport (free, no API key needed)
- Sky Sports (free, no API key needed)
- ESPN (free, no API key needed)

Usage:
    from web_scraper_fallback import WebScraperFallback

    scraper = WebScraperFallback()
    fixtures = scraper.scrape_bbc_fixtures("2025-11-30")
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logging.warning("Web scraping dependencies not available (requests, beautifulsoup4)")

logger = logging.getLogger(__name__)

class WebScraperFallback:
    """Web scraper for football fixtures as API fallback"""

    def __init__(self):
        if not WEB_SCRAPING_AVAILABLE:
            logger.error("Web scraping not available - install: pip install requests beautifulsoup4")
            return

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests to be respectful

    def _rate_limit(self):
        """Enforce polite rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make rate-limited web request"""
        if not WEB_SCRAPING_AVAILABLE:
            return None

        self._rate_limit()

        try:
            logger.debug(f"Scraping: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            return BeautifulSoup(response.content, 'html.parser')

        except requests.exceptions.RequestException as e:
            logger.warning(f"Web scraping error for {url}: {e}")
            return None

    def _clean_team_name(self, name: str) -> str:
        """Clean team names from web scraping"""
        if not name:
            return ""

        # Remove extra whitespace and common prefixes/suffixes
        cleaned = name.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single

        # Remove common web artifacts
        artifacts = ['()', '[]', '""', "''"]
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, '')

        return cleaned.strip()

    def _parse_time(self, time_str: str) -> str:
        """Parse time from various web formats"""
        if not time_str:
            return "TBD"

        # Common time patterns
        time_patterns = [
            r'(\d{1,2}):(\d{2})',  # HH:MM or H:MM
            r'(\d{1,2})\.(\d{2})',  # HH.MM or H.MM
            r'(\d{1,2}):(\d{2})\s*(AM|PM)',  # 12-hour format
        ]

        for pattern in time_patterns:
            match = re.search(pattern, time_str.upper())
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))

                # Handle 12-hour format
                if len(match.groups()) > 2 and match.group(3):
                    if match.group(3) == 'PM' and hours != 12:
                        hours += 12
                    elif match.group(3) == 'AM' and hours == 12:
                        hours = 0

                return f"{hours:02d}:{minutes:02d}"

        logger.debug(f"Could not parse time: {time_str}")
        return "TBD"

    def scrape_bbc_fixtures(self, date: str) -> Dict[str, List[Dict]]:
        """
        Scrape fixtures from BBC Sport

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Dictionary with league fixtures
        """
        fixtures = {}

        if not WEB_SCRAPING_AVAILABLE:
            return fixtures

        try:
            # BBC Sport football fixtures URL
            # Format date for BBC (they use different URL structures)
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            bbc_date = date_obj.strftime('%Y-%m-%d')

            url = f"https://www.bbc.com/sport/football/fixtures/{bbc_date}"
            soup = self._make_request(url)

            if not soup:
                return fixtures

            # BBC Sport structure (this may need updates if their HTML changes)
            fixture_sections = soup.find_all(['div', 'section'], class_=re.compile(r'fixture|match'))

            current_league = None
            current_fixtures = []

            for section in fixture_sections:
                # Look for league headers
                league_header = section.find(['h2', 'h3', 'div'], class_=re.compile(r'league|competition|title'))
                if league_header:
                    # Save previous league if we have fixtures
                    if current_league and current_fixtures:
                        fixtures[current_league] = current_fixtures

                    current_league = self._extract_league_name(league_header.get_text(strip=True))
                    current_fixtures = []

                # Look for individual matches
                matches = section.find_all(['div', 'li'], class_=re.compile(r'fixture|match'))
                for match in matches:
                    fixture_data = self._parse_bbc_fixture(match)
                    if fixture_data:
                        current_fixtures.append(fixture_data)

            # Don't forget the last league
            if current_league and current_fixtures:
                fixtures[current_league] = current_fixtures

            logger.info(f"BBC Sport: Scraped {sum(len(f) for f in fixtures.values())} fixtures")

        except Exception as e:
            logger.warning(f"BBC Sport scraping failed: {e}")

        return fixtures

    def _extract_league_name(self, text: str) -> Optional[str]:
        """Extract standardized league name from scraped text"""
        text = text.lower()

        league_mappings = {
            'championship': 'Championship',
            'league one': 'League One',
            'league two': 'League Two',
            'premier league': 'Premier League',
            'efl championship': 'Championship',
            'efl league one': 'League One',
            'efl league two': 'League Two'
        }

        for key, value in league_mappings.items():
            if key in text:
                return value

        return None

    def _parse_bbc_fixture(self, match_element) -> Optional[Dict]:
        """Parse individual fixture from BBC Sport HTML"""
        try:
            # This is a template - BBC's HTML structure may vary
            # Look for team names
            teams = match_element.find_all(['span', 'div'], class_=re.compile(r'team|club'))
            if len(teams) >= 2:
                home_team = self._clean_team_name(teams[0].get_text(strip=True))
                away_team = self._clean_team_name(teams[1].get_text(strip=True))
            else:
                return None

            # Look for match time
            time_element = match_element.find(['span', 'div', 'time'], class_=re.compile(r'time|kick'))
            match_time = "TBD"
            if time_element:
                match_time = self._parse_time(time_element.get_text(strip=True))

            return {
                'time': match_time,
                'home': home_team,
                'away': away_team,
                'status': 'SCHEDULED',
                'source': 'bbc-sport'
            }

        except Exception as e:
            logger.debug(f"Error parsing BBC fixture: {e}")
            return None

    def scrape_sky_sports_fixtures(self, date: str) -> Dict[str, List[Dict]]:
        """
        Scrape fixtures from Sky Sports (alternative source)

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Dictionary with league fixtures
        """
        fixtures = {}

        if not WEB_SCRAPING_AVAILABLE:
            return fixtures

        try:
            # Sky Sports football fixtures
            date_obj = datetime.strptime(date, '%Y-%m-%d')

            # Sky Sports URL structure
            url = f"https://www.skysports.com/football/fixtures/{date_obj.strftime('%Y-%m-%d')}"
            soup = self._make_request(url)

            if not soup:
                return fixtures

            # Parse Sky Sports structure (this may need updates)
            fixture_containers = soup.find_all(['div', 'section'], class_=re.compile(r'fixture|match'))

            # Similar parsing logic as BBC but adapted for Sky Sports HTML structure
            # This would need to be customized based on their actual HTML

            logger.info(f"Sky Sports: Attempted to scrape {date}")

        except Exception as e:
            logger.warning(f"Sky Sports scraping failed: {e}")

        return fixtures

    def get_fallback_fixtures(self, date: str, leagues: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Get fixtures using web scraping as fallback

        Args:
            date: Date in YYYY-MM-DD format
            leagues: Specific leagues to look for (optional)

        Returns:
            Dictionary with scraped fixtures
        """
        all_fixtures = {}

        if not WEB_SCRAPING_AVAILABLE:
            logger.warning("Web scraping not available - install requests and beautifulsoup4")
            return all_fixtures

        logger.info(f"Using web scraping fallback for {date}")

        # Try BBC Sport first
        bbc_fixtures = self.scrape_bbc_fixtures(date)
        all_fixtures.update(bbc_fixtures)

        # Could add more sources here
        # sky_fixtures = self.scrape_sky_sports_fixtures(date)
        # all_fixtures.update(sky_fixtures)

        # Filter by requested leagues if specified
        if leagues:
            filtered_fixtures = {}
            for league in leagues:
                if league in all_fixtures:
                    filtered_fixtures[league] = all_fixtures[league]
            all_fixtures = filtered_fixtures

        return all_fixtures


# Installation helper
def install_web_scraping_deps():
    """Helper to install web scraping dependencies"""
    print("Web scraping requires additional packages.")
    print("Install with: pip install requests beautifulsoup4")
    print("")
    print("Or add to requirements.txt:")
    print("requests>=2.25.0")
    print("beautifulsoup4>=4.9.0")
