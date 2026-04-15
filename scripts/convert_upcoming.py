#!/usr/bin/env python3

import sys
import json
import re
from datetime import date

# Use today's date unless provided
DATE = sys.argv[1] if len(sys.argv) > 1 else str(date.today())

time_pattern = re.compile(r"^\d{1,2}:\d{2}$")
country_pattern = re.compile(r"^[A-Z ]+:$")

IGNORE = {
    "Standings",
    "Live Standings",
    "Draw",
    "Fixtures",
    "Results"
}

# Maps raw country header -> canonical country name
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

# Maps raw league name (case-insensitive) -> canonical league name
LEAGUE_MAP = {
    # England
    'premier league':       'Premier League',
    'championship':         'Championship',
    'league one':           'League One',
    'league two':           'League Two',
    # France
    'ligue 1':              'Ligue 1',
    'ligue 2':              'Ligue 2',
    # Germany
    'bundesliga':           'Bundesliga',
    '2. bundesliga':        '2. Bundesliga',
    # Italy
    'serie a':              'Serie A',
    'serie b':              'Serie B',
    # Spain
    'la liga':              'La Liga',
    'laliga':               'La Liga',
    'segunda división':     'Segunda División',
    'segunda division':     'Segunda División',
    # Netherlands
    'eredivisie':           'Eredivisie',
    # Portugal
    'primeira liga':        'Primeira Liga',
    'liga portugal':        'Primeira Liga',
    # Scotland
    'scottish premiership': 'Scottish Premiership',
    'scottish championship':'Scottish Championship',
    'scottish league one':  'Scottish League One',
    'scottish league two':  'Scottish League Two',
    # Belgium
    'belgian pro league':   'Belgian Pro League',
    'jupiler pro league':   'Belgian Pro League',
    # Greece
    'super league greece':  'Super League Greece',
    'super league':         'Super League Greece',
    # Turkey
    'süper lig':            'Süper Lig',
    'super lig':            'Süper Lig',
}

data = {}
current_country = None
current_league = None

lines = [l.strip() for l in sys.stdin if l.strip()]

i = 0
while i < len(lines):

    line = lines[i]

    # COUNTRY
    if country_pattern.match(line):
        raw = line.replace(":", "").upper()
        current_country = COUNTRY_MAP.get(raw, raw.title())
        i += 1
        continue

    # LEAGUE
    if line not in IGNORE and not time_pattern.match(line) and not line.isdigit():

        # leagues normally appear before country block
        if i + 1 < len(lines) and lines[i + 1].endswith(":"):
            current_league = LEAGUE_MAP.get(line.lower(), line)
            i += 1
            continue

    # MATCH
    if time_pattern.match(line):

        time = line

        # collect next valid team names
        teams = []
        j = i + 1

        while j < len(lines) and len(teams) < 2:

            t = lines[j]

            if (
                t not in IGNORE
                and not time_pattern.match(t)
                and not t.isdigit()
                and "-" not in t
            ):
                if len(teams) == 0 or t != teams[-1]:
                    teams.append(t)

            j += 1

        if len(teams) == 2 and current_country and current_league:

            if current_country not in data:
                data[current_country] = {}

            if current_league not in data[current_country]:
                data[current_country][current_league] = {
                    "Standings": {},
                    "Fixtures": []
                }

            data[current_country][current_league]["Fixtures"].append({
                "date": DATE,
                "time": time,
                "home": teams[0],
                "away": teams[1]
            })

        i = j
        continue

    i += 1


print(json.dumps(data, indent=2))