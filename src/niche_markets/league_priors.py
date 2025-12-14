"""
League-Specific Priors for Niche Markets

Empirically derived parameters for different leagues, capturing:
- Odd/Even goal bias
- 2nd Half vs 1st Half scoring ratios
- Average goals per match
- Playing style characteristics
"""

LEAGUE_PRIORS = {
    # German Bundesliga
    'D1': {
        'odd_rate': 0.515,
        'half_ratio': 1.35,  # 2nd half / 1st half goals
        'avg_goals': 3.1,
        'style': 'high_intensity',
        'variance': 'medium',
    },

    # German 2. Bundesliga
    'D2': {
        'odd_rate': 0.522,
        'half_ratio': 1.38,
        'avg_goals': 2.9,
        'style': 'physical',
        'variance': 'medium-high',
    },

    # French Ligue 1
    'F1': {
        'odd_rate': 0.510,
        'half_ratio': 1.28,
        'avg_goals': 2.8,
        'style': 'technical',
        'variance': 'medium',
    },

    # French Ligue 2
    'F2': {
        'odd_rate': 0.518,
        'half_ratio': 1.32,
        'avg_goals': 2.7,
        'style': 'balanced',
        'variance': 'medium-high',
    },

    # Italian Serie A
    'I1': {
        'odd_rate': 0.505,
        'half_ratio': 1.22,
        'avg_goals': 2.7,
        'style': 'tactical',
        'variance': 'low',
    },

    # Italian Serie B
    'I2': {
        'odd_rate': 0.520,
        'half_ratio': 1.40,  # High fatigue factor
        'avg_goals': 2.6,
        'style': 'low_fitness',
        'variance': 'high',
    },

    # Spanish La Liga
    'SP1': {
        'odd_rate': 0.512,
        'half_ratio': 1.30,
        'avg_goals': 2.9,
        'style': 'technical',
        'variance': 'medium',
    },

    # Spanish La Liga 2
    'SP2': {
        'odd_rate': 0.525,
        'half_ratio': 1.35,
        'avg_goals': 2.7,
        'style': 'balanced',
        'variance': 'medium-high',
    },

    # Dutch Eredivisie
    'N1': {
        'odd_rate': 0.528,
        'half_ratio': 1.32,
        'avg_goals': 3.2,
        'style': 'attacking',
        'variance': 'high',
    },

    # Dutch Eerste Divisie (KEY LEAGUE - high odd bias)
    'N2': {
        'odd_rate': 0.540,  # Strong odd bias
        'half_ratio': 1.32,
        'avg_goals': 3.0,
        'style': 'open_play',
        'variance': 'very_high',
    },

    # Belgium Pro League
    'B1': {
        'odd_rate': 0.518,
        'half_ratio': 1.30,
        'avg_goals': 2.9,
        'style': 'balanced',
        'variance': 'medium',
    },

    # English Premier League
    'E0': {
        'odd_rate': 0.505,
        'half_ratio': 1.25,
        'avg_goals': 2.9,
        'style': 'balanced',
        'variance': 'medium',
    },

    # English Championship
    'E1': {
        'odd_rate': 0.515,
        'half_ratio': 1.28,
        'avg_goals': 2.7,
        'style': 'physical',
        'variance': 'medium-high',
    },

    # English League One
    'E2': {
        'odd_rate': 0.522,
        'half_ratio': 1.32,
        'avg_goals': 2.8,
        'style': 'direct',
        'variance': 'high',
    },

    # English League Two
    'E3': {
        'odd_rate': 0.530,
        'half_ratio': 1.35,
        'avg_goals': 2.6,
        'style': 'direct',
        'variance': 'high',
    },

    # Scottish Premiership
    'SC0': {
        'odd_rate': 0.518,
        'half_ratio': 1.33,
        'avg_goals': 2.9,
        'style': 'direct',
        'variance': 'medium-high',
    },

    # Scottish Championship
    'SC1': {
        'odd_rate': 0.528,
        'half_ratio': 1.38,
        'avg_goals': 2.8,
        'style': 'physical',
        'variance': 'high',
    },

    # Portuguese Liga
    'P1': {
        'odd_rate': 0.515,
        'half_ratio': 1.28,
        'avg_goals': 2.8,
        'style': 'technical',
        'variance': 'medium',
    },

    # YOUTH leagues (generic - KEY for winning slips)
    'YOUTH': {
        'odd_rate': 0.550,  # Very strong odd bias
        'half_ratio': 1.45,  # Strongest 2nd half bias
        'avg_goals': 3.3,
        'style': 'high_variance',
        'variance': 'extreme',
    },

    # Generic lower divisions
    'LOWER_DIVISION': {
        'odd_rate': 0.530,
        'half_ratio': 1.38,
        'avg_goals': 2.8,
        'style': 'variable',
        'variance': 'high',
    },
}


def get_league_prior(league_code, home_team='', away_team=''):
    """
    Get league-specific priors with automatic youth league detection.

    Args:
        league_code: League identifier (e.g., 'E0', 'N2', 'D1')
        home_team: Home team name (for youth detection)
        away_team: Away team name (for youth detection)

    Returns:
        Dictionary of league priors
    """
    # Youth league detection (case-insensitive)
    youth_keywords = ['YOUTH', 'JONG', 'U23', 'U21', 'U19', 'RESERVE', 'B TEAM']

    combined_names = f"{home_team} {away_team}".upper()
    is_youth = any(keyword in combined_names for keyword in youth_keywords)

    if is_youth:
        return LEAGUE_PRIORS['YOUTH']

    # Return league-specific prior or default to E0 (Premier League)
    return LEAGUE_PRIORS.get(league_code, LEAGUE_PRIORS['E0'])


def calculate_league_priors_from_data(df):
    """
    Calculate empirical priors from historical data.

    Args:
        df: DataFrame with columns FTHG, FTAG, HTHG, HTAG

    Returns:
        Dictionary of calculated priors
    """
    # Total goals
    df['Total_Goals'] = df['FTHG'] + df['FTAG']
    df['Odd_Even'] = df['Total_Goals'].apply(lambda x: 'Odd' if x % 2 == 1 else 'Even')

    # Half goals
    df['1st_Half_Total'] = df['HTHG'] + df['HTAG']
    df['2nd_Half_Total'] = df['Total_Goals'] - df['1st_Half_Total']

    # Calculate priors
    priors = {
        'odd_rate': (df['Odd_Even'] == 'Odd').mean(),
        'even_rate': (df['Odd_Even'] == 'Even').mean(),
        'avg_goals': df['Total_Goals'].mean(),
        'avg_1st_half': df['1st_Half_Total'].mean(),
        'avg_2nd_half': df['2nd_Half_Total'].mean(),
        'half_ratio': df['2nd_Half_Total'].mean() / df['1st_Half_Total'].mean() if df['1st_Half_Total'].mean() > 0 else 1.3,
        '2nd_half_win_rate': (df['2nd_Half_Total'] > df['1st_Half_Total']).mean(),
        '1st_half_win_rate': (df['1st_Half_Total'] > df['2nd_Half_Total']).mean(),
        'equal_halves_rate': (df['1st_Half_Total'] == df['2nd_Half_Total']).mean(),
    }

    return priors


def get_adaptive_prior(league_code, home_team='', away_team='', recent_data=None):
    """
    Get adaptive prior that blends default prior with recent empirical data.

    Args:
        league_code: League identifier
        home_team: Home team name
        away_team: Away team name
        recent_data: Recent match DataFrame (optional)

    Returns:
        Blended prior dictionary
    """
    base_prior = get_league_prior(league_code, home_team, away_team)

    if recent_data is None or len(recent_data) < 10:
        return base_prior

    # Calculate recent empirical priors
    empirical = calculate_league_priors_from_data(recent_data)

    # Blend: 70% base prior, 30% recent empirical
    # (More recent data → increase empirical weight)
    weight_empirical = min(0.4, len(recent_data) / 100)  # Cap at 40%
    weight_base = 1 - weight_empirical

    blended = {
        'odd_rate': weight_base * base_prior['odd_rate'] + weight_empirical * empirical['odd_rate'],
        'half_ratio': weight_base * base_prior['half_ratio'] + weight_empirical * empirical['half_ratio'],
        'avg_goals': weight_base * base_prior['avg_goals'] + weight_empirical * empirical['avg_goals'],
        'style': base_prior['style'],
        'variance': base_prior['variance'],
        'empirical_weight': weight_empirical,
    }

    return blended

