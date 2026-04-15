#!/usr/bin/env python3
"""
Convert raw fixtures.txt (pasted from webpage) to data/raw/upcomingMatches.json
Format: "COUNTRY: League Name" header, then times, then team names (each repeated twice)
Usage: cat scripts/fixtures.txt | python3 scripts/convert_fixtures_to_json.py [DATE]
"""
import sys
import json
import re
import os
from datetime import date

DATE = sys.argv[1] if len(sys.argv) > 1 else str(date.today())

COUNTRY_MAP = {
    'ENGLAND':     'England',
    'FRANCE':      'France',
    'GERMANY':     'Germany',
    'ITALY':       'Italy',
    'SPAIN':       'Spain',
    'NETHERLANDS': 'Netherlands',
    'PORTUGAL':    'Portugal',
    'SCOTLAND':    'Scotland',
    'BELGIUM':     'Belgium',
    'GREECE':      'Greece',
    'TURKEY':      'Turkey',
}

LEAGUE_MAP = {
    'premier league':        'Premier League',
    'championship':          'Championship',
    'league one':            'League One',
    'league two':            'League Two',
    'ligue 1':               'Ligue 1',
    'ligue 2':               'Ligue 2',
    'bundesliga':            'Bundesliga',
    '2. bundesliga':         '2. Bundesliga',
    'serie a':               'Serie A',
    'serie b':               'Serie B',
    'la liga':               'La Liga',
    'laliga':                'La Liga',
    'segunda división':      'Segunda División',
    'segunda division':      'Segunda División',
    'eredivisie':            'Eredivisie',
    'primeira liga':         'Primeira Liga',
    'liga portugal':         'Primeira Liga',
    'scottish premiership':  'Scottish Premiership',
    'scottish championship': 'Scottish Championship',
    'scottish league one':   'Scottish League One',
    'scottish league two':   'Scottish League Two',
    'belgian pro league':    'Belgian Pro League',
    'jupiler pro league':    'Belgian Pro League',
    'super league greece':   'Super League Greece',
    'super league':          'Super League Greece',
    'süper lig':             'Süper Lig',
    'super lig':             'Süper Lig',
}

# Lines to discard entirely
NOISE = {
    'standings', 'live standings', 'draw', 'fixtures', 'results',
    'preview', 'live>', 't', '-', ''
}

time_re = re.compile(r'^\d{1,2}:\d{2}$')
# Matches "ENGLAND: League One" or "ENGLAND:" (league part optional)
header_re = re.compile(r'^([A-Z][A-Z ]+):\s*(.*)?$')


def canonical_league(raw):
    return LEAGUE_MAP.get(raw.strip().lower(), raw.strip().title())


def parse(lines):
    lines = [l.strip() for l in lines]

    # Split into blocks on each "COUNTRY: ..." header line
    blocks = []  # list of (country, league, [lines])
    current = None

    for line in lines:
        m = header_re.match(line)
        if m and m.group(1).upper() in COUNTRY_MAP:
            if current:
                blocks.append(current)
            country = COUNTRY_MAP[m.group(1).upper()]
            league_raw = m.group(2).strip() if m.group(2) else ''
            league = canonical_league(league_raw) if league_raw else None
            current = (country, league, [])
        elif current is not None:
            current[2].append(line)

    if current:
        blocks.append(current)

    data = {}

    for country, league, content in blocks:
        # Collect times and team names separately, discarding noise
        times = []
        teams = []

        for line in content:
            if not line or line.lower() in NOISE:
                continue
            if time_re.match(line):
                times.append(line)
            elif not line.isdigit():
                teams.append(line)

        # Teams are listed as: Home Home Away Away Home Home Away Away ...
        # Deduplicate consecutive duplicates to get [Home, Away, Home, Away ...]
        deduped = []
        for t in teams:
            if not deduped or t != deduped[-1]:
                deduped.append(t)

        # Pair up: deduped[0]=home, deduped[1]=away per match
        matches = []
        for i, time in enumerate(times):
            home_idx = i * 2
            away_idx = i * 2 + 1
            if away_idx < len(deduped):
                matches.append({
                    'date': DATE,
                    'time': time,
                    'home': deduped[home_idx],
                    'away': deduped[away_idx],
                })

        if matches and league:
            data.setdefault(country, {})
            data[country].setdefault(league, {'Standings': {}, 'Fixtures': []})
            data[country][league]['Fixtures'].extend(matches)

    return data


if __name__ == '__main__':
    lines = sys.stdin.read().splitlines()
    result = parse(lines)
    out_path = 'data/raw/upcomingMatches.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    total = sum(len(v['Fixtures']) for c in result.values() for v in c.values())
    leagues = sum(len(v) for v in result.values())
    print(f"✅ Written {total} matches across {leagues} leagues → {out_path}")
