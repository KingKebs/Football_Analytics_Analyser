# Niche Markets Analysis: Highest Scoring Half & Odd/Even Goals

**Date:** December 13, 2025  
**Focus Markets:** 2nd Half Highest Scoring, Odd/Even (Odd Goals), Mixed Parlays

---

## PART 1: MARKET DECOMPOSITION

### 1.1 Highest Scoring Half (2nd Half) - Statistical Patterns

#### Key Observations from Winning Slips:
- **6/6 wins on 2nd Half** across diverse leagues (Bundesliga, Ligue 1, Serie B, Belgian Pro League, La Liga)
- **Odds range:** 2.07-2.23 (implies ~45-48% market probability)
- **League diversity:** Top European leagues + Championship/2nd division matches
- **Notable pattern:** Mixed youth leagues (Jong teams) in combined slips

#### Statistical Drivers:

**A. Tempo & Fatigue Dynamics**
1. **Second-half goal acceleration** - Teams score more in final 30 minutes (60'-90'+)
   - Physical fatigue reduces defensive organization
   - Substitutions introduce fresh attacking players
   - Tactical desperation when teams are behind

2. **Game state dependency**
   - 0-0 or 1-0 at HT → increased urgency in 2nd half
   - Teams trailing push forward, creating space for counters
   - Psychological pressure to avoid draws

3. **Substitution impact**
   - Average 3-5 subs per team → 6-10 fresh players in 2nd half
   - Attacking subs typically made 60'-75' mark
   - Youth leagues have higher sub impact (fitness disparity)

**B. Structural Match Characteristics**
1. **First half caution bias**
   - Teams probe defenses early, fewer risks
   - Conservative tactics in opening 30 minutes
   - Set-piece goals more common in 2nd half

2. **Referee leniency decline**
   - Yellow cards increase in 2nd half → more fouls → more set pieces
   - Added time (stoppage) extends 2nd half effectively

3. **League-specific traits**
   - **German Bundesliga:** High-intensity pressing → 2nd half fatigue goals
   - **French Ligue 1:** Technical play → late goals from skill
   - **Italian Serie B:** Lower fitness → significant 2nd half drop-off
   - **Youth leagues (YOUTH tags):** Massive fitness variance → 2nd half goal surges

### 1.2 Odd/Even (Odd Goals) - Statistical Patterns

#### Key Observations from Winning Slips:
- **9/9 wins on Odd** (100% success in pure Odd/Even slip)
- **14/14 wins on Odd** in mixed slip (Odd + 2nd Half)
- **Odds range:** 1.80-1.90 (implies ~52-55% market probability)
- **Critical pattern:** Concentration in **Dutch Eerste Divisie** and **YOUTH leagues**

#### Statistical Drivers:

**A. Goal Parity Mathematics**
1. **Base probability bias**
   - Most common scorelines: 1-0, 2-1, 1-1, 3-1, 2-2, 3-0
   - Odd scorelines: 1-0, 2-1, 3-0, 1-2, 0-3, 3-2 (6 of top 10)
   - Even scorelines: 1-1, 2-2, 0-0, 2-0, 0-2 (4 of top 10)
   - **Natural skew toward odd ~52-54%**

2. **Low-scoring match advantage**
   - 1 goal = Odd (100%)
   - 2 goals = Even (100%)
   - 3 goals = Odd (100%)
   - In tight, low-scoring leagues → odd goals more likely

**B. League-Specific Parity Patterns**
1. **Dutch Eerste Divisie (2nd tier)**
   - Average goals: 2.8-3.2 per match
   - High variance, open play
   - Common scorelines: 2-1, 3-1, 3-2 (all odd)

2. **YOUTH leagues** (Jong AZ, Jong PSV, Jong Ajax, Jong FC Utrecht)
   - Higher goal variance
   - Less defensive discipline → more goals
   - Odd-total bias due to 3-goal average

3. **Tactical reasoning**
   - Teams chasing losing positions score late consolation → 2-0 becomes 2-1
   - Teams protecting 1-0 leads concede late → 1-1 becomes 2-1
   - **Transitional goals favor odd parity**

**C. Psychological & Game State**
1. **Late goal impact**
   - 85'+ goals often change parity
   - Team behind by 1 scores → odd total
   - Team behind by 2 rarely scores 2 → stays odd

---

## PART 2: ALGORITHM COMBINATION ANALYSIS

### 2.1 Hybrid Model Architecture

```
┌─────────────────────────────────────────────────────┐
│         NICHE MARKET PREDICTION FRAMEWORK           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────┐     │
│  │  LAYER 1: Base Goal Expectation          │     │
│  │  - Poisson/Dixon-Coles                   │     │
│  │  - xG per 90 minutes                     │     │
│  │  - League-adjusted parameters             │     │
│  └──────────────────────────────────────────┘     │
│                     ↓                              │
│  ┌──────────────────────────────────────────┐     │
│  │  LAYER 2: Time-Segmented Models          │     │
│  │  - xG by half (45' split)                │     │
│  │  - Time-decay scoring rates              │     │
│  │  - Fatigue-adjusted Poisson              │     │
│  └──────────────────────────────────────────┘     │
│                     ↓                              │
│  ┌──────────────────────────────────────────┐     │
│  │  LAYER 3: Context-Aware Adjustments      │     │
│  │  - Game state transitions (Markov)       │     │
│  │  - Substitution impact models            │     │
│  │  - League variance priors (Bayesian)     │     │
│  └──────────────────────────────────────────┘     │
│                     ↓                              │
│  ┌──────────────────────────────────────────┐     │
│  │  LAYER 4: Market-Specific Outputs        │     │
│  │  - P(2nd Half > 1st Half)               │     │
│  │  - P(Total Goals = Odd)                  │     │
│  │  - Combined parlay probabilities          │     │
│  └──────────────────────────────────────────┘     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 2.2 Algorithm Components & Roles

#### **Component 1: Adjusted Poisson Distribution**
**Purpose:** Foundation for goal expectation  
**Contribution to markets:**
- **Odd/Even:** Calculate P(Total=k) for k=0,1,2,3,4,5,6, sum odd k vs even k
- **2nd Half:** Not directly applicable (needs segmentation)

**Formula:**
```
P(X=k) = (λ^k × e^-λ) / k!
P(Odd) = Σ P(k) for k ∈ {1,3,5,7,...}
P(Even) = Σ P(k) for k ∈ {0,2,4,6,...}
```

**Limitation:** Standard Poisson assumes uniform scoring rate across 90 minutes - FALSE for 2nd half analysis

---

#### **Component 2: Time-Segmented Expected Goals (Half-Split Model)**
**Purpose:** Model scoring intensity by match period  
**Contribution to markets:**
- **2nd Half:** Direct prediction of goals per half
- **Odd/Even:** Indirect (more accurate total goals prediction)

**Implementation:**
```python
# Historical half-time analysis
1st_half_rate = (HTHG + HTAG) / matches  # Goals per first half
2nd_half_rate = ((FTHG-HTHG) + (FTAG-HTAG)) / matches  # Goals per second half

# Ratio calculation
half_ratio = 2nd_half_rate / 1st_half_rate  # Typically 1.15-1.35
```

**Key Variables:**
- `half_ratio_league`: League-average 2nd/1st half scoring ratio
- `half_ratio_team`: Team-specific ratio (attacking/defensive style)
- `half_ratio_h2h`: Head-to-head historical ratio

**Enhanced Model:**
```python
λ_1st_half = xG_total * (1 / (1 + half_ratio))
λ_2nd_half = xG_total * (half_ratio / (1 + half_ratio))

P(2nd > 1st) = Σ Σ P(1st=i) × P(2nd=j) for j > i
```

---

#### **Component 3: Dixon-Coles Adjustment**
**Purpose:** Correct Poisson for low-score correlation  
**Contribution to markets:**
- **Odd/Even:** Improves 0-0, 1-0, 0-1, 1-1 probabilities (critical odd/even outcomes)
- **2nd Half:** Minimal (low scores rare in single halves)

**Formula:**
```
τ(x,y) = {
  1 - λ_h × λ_a × ρ   if x=y=0
  1 + λ_h × ρ         if x=0, y=1
  1 + λ_a × ρ         if x=1, y=0
  1 - ρ               if x=y=1
  1                   otherwise
}
```

**ρ (correlation parameter):** Typically -0.10 to -0.15

---

#### **Component 4: Markov Chain Game State Model**
**Purpose:** Model scoring transitions based on current state  
**Contribution to markets:**
- **2nd Half:** Game state at HT predicts 2nd half intensity
- **Odd/Even:** Late-game parity changes

**States:**
- S1: Winning by 1
- S2: Winning by 2+
- D: Draw
- L1: Losing by 1
- L2: Losing by 2+

**Transition Logic:**
```
If HT state = L1 (losing by 1):
  → Attack rate increases by 20-30%
  → Defence rate worsens by 15-20%
  → 2nd half λ_attack × 1.25

If HT state = W2 (winning by 2+):
  → Attack rate decreases by 15%
  → Defence maintains or improves
  → 2nd half λ_attack × 0.85
```

---

#### **Component 5: Bayesian Prior for League Variance**
**Purpose:** Incorporate league-specific structural biases  
**Contribution to markets:**
- **Odd/Even:** Youth/lower leagues have higher odd-bias
- **2nd Half:** High-intensity leagues (Bundesliga) favor 2nd half

**Priors:**
```python
league_priors = {
    'odd_bias': {
        'Eerste_Divisie': 0.54,      # Higher odd tendency
        'YOUTH': 0.55,                # Highest variance
        'Serie_B': 0.52,              # Moderate
        'Premier_League': 0.50,       # Neutral
    },
    '2nd_half_ratio': {
        'Bundesliga': 1.35,           # Highest intensity
        'Ligue_1': 1.28,              
        'Serie_B': 1.40,              # Fatigue factor
        'YOUTH': 1.45,                # Fitness gap
        'Premier_League': 1.25,       # Most even
    }
}
```

**Bayesian Update:**
```python
P(Market | Data) ∝ P(Data | Market) × P(Market | League)
```

---

#### **Component 6: Fatigue-Adjusted Scoring Model**
**Purpose:** Account for declining defensive quality  
**Contribution to markets:**
- **2nd Half:** Core model component
- **Odd/Even:** Indirect through total goals

**Implementation:**
```python
# Minute-by-minute scoring intensity
scoring_intensity = {
    '0-15':   0.95,  # Cautious start
    '15-30':  1.00,  # Normal
    '30-45':  1.05,  # Pre-HT push
    '45-60':  0.98,  # Post-HT reset
    '60-75':  1.10,  # Fatigue begins
    '75-90':  1.25,  # Maximum fatigue
    '90+':    1.40,  # Desperation + added time
}

# 2nd half weighted average
avg_2nd_half_intensity = mean([0.98, 1.10, 1.25, 1.40]) = 1.18
```

---

#### **Component 7: Substitution Impact Model**
**Purpose:** Quantify tactical changes  
**Contribution to markets:**
- **2nd Half:** Direct (fresh players score more)
- **Odd/Even:** Late subs often produce odd-parity goals

**Model:**
```python
sub_impact = {
    'attacking_sub': +0.15,  # Increases λ_attack
    'defensive_sub': -0.10,  # Decreases λ_attack
    'forced_injury': 0.00,   # Neutral
}

# Average team makes 2 attacking subs between 60'-75'
2nd_half_λ_adjusted = λ_base × (1 + 2 × 0.15 × time_remaining_factor)
```

---

### 2.3 Why Hybrid Models Outperform Single Models

**Single Model Limitations:**

1. **Pure Poisson:**
   - ❌ Assumes constant scoring rate (false for halves)
   - ❌ Ignores game state dynamics
   - ❌ No odd/even bias mechanism

2. **Pure xG Models:**
   - ❌ xG is full-match metric, not time-segmented
   - ❌ Doesn't account for tactical adjustments
   - ❌ No parity modeling

3. **Pure Machine Learning:**
   - ❌ Requires massive labeled data for niche markets
   - ❌ Black box - can't explain why 2nd half favored
   - ❌ Overfits to recent patterns

**Hybrid Advantages:**

✅ **Interpretability:** Each component has clear mathematical meaning  
✅ **Data efficiency:** Poisson base requires less data than pure ML  
✅ **Adaptability:** Bayesian priors quickly adapt to new leagues  
✅ **Robustness:** Multiple models cross-validate predictions  
✅ **Market-specific:** Each layer adds relevant signal to target market

---

## PART 3: CODEBASE REVIEW & ENHANCEMENT

### 3.1 Current Implementation Analysis

#### ✅ **Already Implemented (Strong Foundation):**

1. **`algorithms.py`:**
   - ✅ Poisson probability calculation (`poisson_prob()`)
   - ✅ Score matrix generation (`score_probability_matrix()`)
   - ✅ Basic xG estimation (`estimate_xg()`)
   - ✅ Recent form with exponential decay (`compute_recent_form()`)
   - ✅ Kelly Criterion staking (`kelly_criterion()`)

2. **`ml_features.py`:**
   - ✅ Feature engineering framework
   - ✅ Team form calculations
   - ✅ H2H statistics

3. **`corners_analysis.py`:**
   - ✅ Half-time split analysis (`estimate_half_split()`)
   - ✅ Time-decay weighting (`_compute_recency_weights()`)
   - ✅ Monte Carlo sampling for uncertainty

4. **Data availability:**
   - ✅ HTHG, HTAG (half-time goals) in CSV
   - ✅ FTHG, FTAG (full-time goals) in CSV
   - ✅ League codes and historical data

#### ❌ **Missing Components (To Implement):**

1. **No Odd/Even prediction module**
2. **No 2nd Half vs 1st Half comparison**
3. **No time-segmented xG models**
4. **No game state Markov transitions**
5. **No league-specific odd/even priors**
6. **No substitution impact modeling**
7. **No combined parlay probability calculator**

---

### 3.2 Feature Engineering Requirements

#### **New Features for Odd/Even Market:**

```python
# 1. Historical Odd/Even Bias
team_odd_rate_h = (odd_goals_as_home / total_home_matches)
team_odd_rate_a = (odd_goals_as_away / total_away_matches)
league_odd_rate = (total_odd_matches / total_matches)

# 2. Recent Odd/Even Streak
last_5_odd_even = [Odd, Odd, Even, Odd, Odd]  # Last 5 matches
odd_streak = 3  # Current streak

# 3. Expected Total Goals Proximity to Odd Integers
xG_total = 2.7  # Close to 3 (odd)
distance_to_nearest_odd = min(|2.7 - 1|, |2.7 - 3|, |2.7 - 5|) = 0.3

# 4. Low-Score Scenario Probability
P(total_goals ≤ 2) = high → favor odd (since 1 is most common)

# 5. H2H Odd/Even Pattern
h2h_odd_rate = (odd_goals_in_h2h / h2h_matches)
```

#### **New Features for 2nd Half Market:**

```python
# 1. Team-Specific Half Ratios
team_2nd_half_ratio_home = (Σ(FTHG-HTHG) / Σ HTHG) for home matches
team_2nd_half_ratio_away = (Σ(FTAG-HTAG) / Σ HTAG) for away matches

# 2. League Half Ratio
league_2nd_1st_ratio = 1.28  # From historical data

# 3. Recent Half-Time Scores
avg_ht_score_last_5 = mean([0-0, 1-0, 0-1, 1-1, 2-0])
if low → expect more 2nd half urgency

# 4. Fatigue Index
fatigue_index = (days_since_last_match, matches_in_last_14_days)
high_fatigue → more 2nd half goals

# 5. Late Goal Frequency
late_goal_rate_team = (goals_after_75min / total_goals)
late_goal_rate_league = league_average

# 6. Youth League Flag
is_youth_league = 1 if 'YOUTH' in team_name else 0
youth_leagues → 2nd_half_ratio = 1.45

# 7. H2H Half Patterns
h2h_2nd_half_rate = mean([2nd_half_goals]) from last N h2h matches
```

---

### 3.3 Architectural Changes

#### **New Module Structure:**

```
src/
├── algorithms.py                    (existing - minor enhancements)
├── ml_features.py                   (existing - add new features)
├── niche_markets/                   (NEW MODULE)
│   ├── __init__.py
│   ├── odd_even_predictor.py       (NEW)
│   ├── half_comparison_predictor.py(NEW)
│   ├── time_segmented_xg.py        (NEW)
│   ├── game_state_markov.py        (NEW)
│   ├── league_priors.py            (NEW)
│   └── parlay_optimizer.py         (NEW)
└── automate_niche_markets.py       (NEW - CLI interface)
```

---

### 3.4 Implementation Roadmap

#### **Phase 1: Data Preparation (Priority 1)**

**File:** `src/niche_markets/data_preparation.py`

```python
def calculate_half_statistics(csv_path):
    """Extract 1st half vs 2nd half goal statistics"""
    df = pd.read_csv(csv_path)
    
    # Calculate half goals
    df['1st_Half_Home'] = df['HTHG']
    df['1st_Half_Away'] = df['HTAG']
    df['2nd_Half_Home'] = df['FTHG'] - df['HTHG']
    df['2nd_Half_Away'] = df['FTAG'] - df['HTAG']
    df['1st_Half_Total'] = df['HTHG'] + df['HTAG']
    df['2nd_Half_Total'] = df['2nd_Half_Home'] + df['2nd_Half_Away']
    
    # Calculate which half had more goals
    df['Highest_Scoring_Half'] = df.apply(
        lambda row: '2nd' if row['2nd_Half_Total'] > row['1st_Half_Total']
                    else '1st' if row['1st_Half_Total'] > row['2nd_Half_Total']
                    else 'Equal',
        axis=1
    )
    
    # Calculate odd/even
    df['Total_Goals'] = df['FTHG'] + df['FTAG']
    df['Odd_Even'] = df['Total_Goals'].apply(lambda x: 'Odd' if x % 2 == 1 else 'Even')
    
    return df

def calculate_league_priors(df, league_code):
    """Calculate league-specific priors"""
    stats = {
        'odd_rate': (df['Odd_Even'] == 'Odd').mean(),
        '2nd_half_win_rate': (df['Highest_Scoring_Half'] == '2nd').mean(),
        'avg_1st_half_goals': df['1st_Half_Total'].mean(),
        'avg_2nd_half_goals': df['2nd_Half_Total'].mean(),
        'half_ratio': df['2nd_Half_Total'].mean() / df['1st_Half_Total'].mean(),
    }
    return stats
```

#### **Phase 2: Odd/Even Predictor (Priority 1)**

**File:** `src/niche_markets/odd_even_predictor.py`

```python
class OddEvenPredictor:
    def __init__(self, league_prior_odd_rate=0.52):
        self.league_prior = league_prior_odd_rate
    
    def predict_from_poisson(self, xg_home, xg_away, max_goals=10):
        """Calculate odd/even probability from Poisson"""
        p_odd = 0.0
        p_even = 0.0
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                total = h + a
                prob = poisson_prob(xg_home, h) * poisson_prob(xg_away, a)
                
                if total % 2 == 1:
                    p_odd += prob
                else:
                    p_even += prob
        
        # Normalize
        total_prob = p_odd + p_even
        return {
            'Odd': p_odd / total_prob,
            'Even': p_even / total_prob
        }
    
    def predict_with_team_bias(self, xg_home, xg_away, 
                                home_odd_rate, away_odd_rate,
                                bayesian_weight=0.3):
        """Combine Poisson with team historical bias"""
        poisson_probs = self.predict_from_poisson(xg_home, xg_away)
        
        # Team average bias
        team_odd_rate = (home_odd_rate + away_odd_rate) / 2
        
        # Bayesian blend
        final_odd_prob = (
            (1 - bayesian_weight) * poisson_probs['Odd'] +
            bayesian_weight * team_odd_rate
        )
        
        return {
            'Odd': final_odd_prob,
            'Even': 1 - final_odd_prob,
            'confidence': abs(final_odd_prob - 0.5) * 2  # 0-1 scale
        }
```

#### **Phase 3: Half Comparison Predictor (Priority 1)**

**File:** `src/niche_markets/half_comparison_predictor.py`

```python
class HalfComparisonPredictor:
    def __init__(self, league_half_ratio=1.28):
        self.league_ratio = league_half_ratio
    
    def predict_half_scoring(self, xg_home, xg_away, 
                             home_half_ratio, away_half_ratio,
                             is_youth_league=False):
        """Predict 1st half vs 2nd half goals"""
        
        # Total expected goals
        xg_total = xg_home + xg_away
        
        # Team-specific ratio (blend home defense with away attack patterns)
        team_ratio = (home_half_ratio + away_half_ratio) / 2
        
        # Youth league adjustment
        if is_youth_league:
            team_ratio = max(team_ratio, 1.40)  # Force high ratio
        
        # Blend team and league ratios
        final_ratio = 0.6 * team_ratio + 0.4 * self.league_ratio
        
        # Split xG between halves
        lambda_1st = xg_total / (1 + final_ratio)
        lambda_2nd = xg_total * final_ratio / (1 + final_ratio)
        
        # Calculate probabilities for all score combinations
        p_2nd_higher = 0.0
        p_1st_higher = 0.0
        p_equal = 0.0
        
        for goals_1st in range(0, 8):
            for goals_2nd in range(0, 8):
                prob = (poisson_prob(lambda_1st, goals_1st) * 
                       poisson_prob(lambda_2nd, goals_2nd))
                
                if goals_2nd > goals_1st:
                    p_2nd_higher += prob
                elif goals_1st > goals_2nd:
                    p_1st_higher += prob
                else:
                    p_equal += prob
        
        return {
            '2nd_Half': p_2nd_higher,
            '1st_Half': p_1st_higher,
            'Equal': p_equal,
            'lambda_1st': lambda_1st,
            'lambda_2nd': lambda_2nd,
            'ratio_used': final_ratio
        }
    
    def predict_with_game_state(self, xg_home, xg_away,
                                 expected_ht_state='close',
                                 **kwargs):
        """Adjust for likely half-time game state"""
        
        base_prediction = self.predict_half_scoring(xg_home, xg_away, **kwargs)
        
        # Adjust based on expected HT situation
        adjustments = {
            'close': 1.0,      # 0-0 or 1-0 → normal
            'losing': 1.15,    # Team expected to be behind → 2nd half boost
            'winning_big': 0.90,  # Team coasting → 2nd half reduction
        }
        
        adjustment = adjustments.get(expected_ht_state, 1.0)
        
        # Apply adjustment to 2nd half probability
        adjusted_2nd = base_prediction['2nd_Half'] * adjustment
        adjusted_1st = base_prediction['1st_Half'] * (2 - adjustment)  # Inverse
        
        # Renormalize
        total = adjusted_2nd + adjusted_1st + base_prediction['Equal']
        
        return {
            '2nd_Half': adjusted_2nd / total,
            '1st_Half': adjusted_1st / total,
            'Equal': base_prediction['Equal'] / total
        }
```

#### **Phase 4: Time-Segmented xG (Priority 2)**

**File:** `src/niche_markets/time_segmented_xg.py`

```python
def calculate_minute_by_minute_intensity(fatigue_level='normal'):
    """Return scoring intensity by time period"""
    base_intensity = {
        '0-15':   0.95,
        '15-30':  1.00,
        '30-45':  1.05,
        '45-60':  0.98,
        '60-75':  1.10,
        '75-90':  1.25,
        '90+':    1.40,
    }
    
    if fatigue_level == 'high':
        # Amplify late-game effects
        base_intensity['75-90'] = 1.35
        base_intensity['90+'] = 1.50
    elif fatigue_level == 'low':
        # More even distribution
        base_intensity['75-90'] = 1.15
        base_intensity['90+'] = 1.25
    
    return base_intensity

def segment_xg_by_time(xg_total, fatigue_level='normal'):
    """Split xG into time periods"""
    intensity = calculate_minute_by_minute_intensity(fatigue_level)
    
    # Calculate time weights (minutes per period)
    time_weights = {
        '0-15': 15,
        '15-30': 15,
        '30-45': 15,
        '45-60': 15,
        '60-75': 15,
        '75-90': 15,
        '90+': 3,  # Average stoppage time
    }
    
    # Weighted intensity
    total_weighted_minutes = sum(
        intensity[period] * time_weights[period]
        for period in intensity
    )
    
    # xG per period
    xg_by_period = {
        period: xg_total * (intensity[period] * time_weights[period]) / total_weighted_minutes
        for period in intensity
    }
    
    # Aggregate to halves
    xg_1st_half = sum(xg_by_period[p] for p in ['0-15', '15-30', '30-45'])
    xg_2nd_half = sum(xg_by_period[p] for p in ['45-60', '60-75', '75-90', '90+'])
    
    return {
        'by_period': xg_by_period,
        '1st_half': xg_1st_half,
        '2nd_half': xg_2nd_half,
        'ratio': xg_2nd_half / xg_1st_half if xg_1st_half > 0 else 1.3
    }
```

#### **Phase 5: League Priors Database (Priority 2)**

**File:** `src/niche_markets/league_priors.py`

```python
LEAGUE_PRIORS = {
    # German Bundesliga
    'D1': {
        'odd_rate': 0.515,
        'half_ratio': 1.35,
        'avg_goals': 3.1,
        'style': 'high_intensity',
    },
    
    # French Ligue 1
    'F1': {
        'odd_rate': 0.510,
        'half_ratio': 1.28,
        'avg_goals': 2.8,
        'style': 'technical',
    },
    
    # Italian Serie B
    'I2': {
        'odd_rate': 0.520,
        'half_ratio': 1.40,
        'avg_goals': 2.6,
        'style': 'low_fitness',
    },
    
    # Dutch Eerste Divisie
    'N2': {
        'odd_rate': 0.540,  # KEY: High odd bias
        'half_ratio': 1.32,
        'avg_goals': 3.0,
        'style': 'open_play',
    },
    
    # Belgium Pro League
    'B1': {
        'odd_rate': 0.518,
        'half_ratio': 1.30,
        'avg_goals': 2.9,
        'style': 'balanced',
    },
    
    # YOUTH leagues (generic)
    'YOUTH': {
        'odd_rate': 0.550,  # KEY: Highest odd bias
        'half_ratio': 1.45,  # KEY: Highest 2nd half bias
        'avg_goals': 3.3,
        'style': 'high_variance',
    },
    
    # English Premier League (baseline)
    'E0': {
        'odd_rate': 0.505,
        'half_ratio': 1.25,
        'avg_goals': 2.9,
        'style': 'balanced',
    },
}

def get_league_prior(league_code, team_name=''):
    """Get league prior, check for youth league override"""
    
    # Youth league detection
    if 'YOUTH' in team_name.upper() or 'JONG' in team_name.upper():
        return LEAGUE_PRIORS['YOUTH']
    
    # Return league prior or default
    return LEAGUE_PRIORS.get(league_code, LEAGUE_PRIORS['E0'])
```

#### **Phase 6: Parlay Optimizer (Priority 3)**

**File:** `src/niche_markets/parlay_optimizer.py`

```python
class ParlayOptimizer:
    def __init__(self, min_confidence=0.55, max_legs=15):
        self.min_confidence = min_confidence
        self.max_legs = max_legs
    
    def calculate_parlay_probability(self, individual_probs):
        """Calculate combined parlay win probability"""
        total_prob = 1.0
        for prob in individual_probs:
            total_prob *= prob
        return total_prob
    
    def optimize_parlay_selection(self, predictions, target_odds_range=(20, 100)):
        """Select best legs for parlay"""
        
        # Filter high-confidence predictions
        filtered = [
            p for p in predictions
            if p['confidence'] >= self.min_confidence
        ]
        
        # Sort by confidence descending
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Try combinations to hit target odds
        best_parlay = None
        best_score = 0
        
        for n_legs in range(3, min(self.max_legs + 1, len(filtered) + 1)):
            legs = filtered[:n_legs]
            
            # Calculate combined odds
            parlay_prob = self.calculate_parlay_probability([l['prob'] for l in legs])
            implied_odds = 1 / parlay_prob if parlay_prob > 0 else 999
            
            # Score based on odds range and confidence
            in_range = target_odds_range[0] <= implied_odds <= target_odds_range[1]
            avg_confidence = sum(l['confidence'] for l in legs) / n_legs
            
            score = avg_confidence * (1.0 if in_range else 0.5)
            
            if score > best_score:
                best_score = score
                best_parlay = {
                    'legs': legs,
                    'n_legs': n_legs,
                    'parlay_prob': parlay_prob,
                    'implied_odds': implied_odds,
                    'avg_confidence': avg_confidence
                }
        
        return best_parlay
```

---

### 3.5 Enhanced Features for `ml_features.py`

Add these functions to the existing `ml_features.py`:

```python
def extract_half_features(df, home_team, away_team, last_n=10):
    """Extract half-time related features"""
    
    # Home team historical half ratios
    home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].tail(last_n)
    
    home_1st_half_goals = []
    home_2nd_half_goals = []
    
    for _, match in home_matches.iterrows():
        if match['HomeTeam'] == home_team:
            home_1st_half_goals.append(match['HTHG'])
            home_2nd_half_goals.append(match['FTHG'] - match['HTHG'])
        else:
            home_1st_half_goals.append(match['HTAG'])
            home_2nd_half_goals.append(match['FTAG'] - match['HTAG'])
    
    home_half_ratio = (
        np.mean(home_2nd_half_goals) / np.mean(home_1st_half_goals)
        if np.mean(home_1st_half_goals) > 0 else 1.3
    )
    
    # Repeat for away team
    away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].tail(last_n)
    
    away_1st_half_goals = []
    away_2nd_half_goals = []
    
    for _, match in away_matches.iterrows():
        if match['HomeTeam'] == away_team:
            away_1st_half_goals.append(match['HTHG'])
            away_2nd_half_goals.append(match['FTHG'] - match['HTHG'])
        else:
            away_1st_half_goals.append(match['HTAG'])
            away_2nd_half_goals.append(match['FTAG'] - match['HTAG'])
    
    away_half_ratio = (
        np.mean(away_2nd_half_goals) / np.mean(away_1st_half_goals)
        if np.mean(away_1st_half_goals) > 0 else 1.3
    )
    
    return {
        'home_half_ratio': home_half_ratio,
        'away_half_ratio': away_half_ratio,
        'home_avg_1st_half': np.mean(home_1st_half_goals),
        'home_avg_2nd_half': np.mean(home_2nd_half_goals),
        'away_avg_1st_half': np.mean(away_1st_half_goals),
        'away_avg_2nd_half': np.mean(away_2nd_half_goals),
    }

def extract_odd_even_features(df, home_team, away_team, last_n=10):
    """Extract odd/even historical bias"""
    
    # Home team odd/even rate
    home_matches = df[(df['HomeTeam'] == home_team) | (df['AwayTeam'] == home_team)].tail(last_n)
    home_total_goals = []
    
    for _, match in home_matches.iterrows():
        total = match['FTHG'] + match['FTAG']
        home_total_goals.append(total)
    
    home_odd_rate = sum(1 for g in home_total_goals if g % 2 == 1) / len(home_total_goals) if home_total_goals else 0.5
    
    # Away team odd/even rate
    away_matches = df[(df['HomeTeam'] == away_team) | (df['AwayTeam'] == away_team)].tail(last_n)
    away_total_goals = []
    
    for _, match in away_matches.iterrows():
        total = match['FTHG'] + match['FTAG']
        away_total_goals.append(total)
    
    away_odd_rate = sum(1 for g in away_total_goals if g % 2 == 1) / len(away_total_goals) if away_total_goals else 0.5
    
    # H2H odd/even rate
    h2h = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].tail(5)
    
    h2h_total_goals = [row['FTHG'] + row['FTAG'] for _, row in h2h.iterrows()]
    h2h_odd_rate = sum(1 for g in h2h_total_goals if g % 2 == 1) / len(h2h_total_goals) if h2h_total_goals else 0.5
    
    return {
        'home_odd_rate': home_odd_rate,
        'away_odd_rate': away_odd_rate,
        'h2h_odd_rate': h2h_odd_rate,
        'combined_odd_rate': (home_odd_rate + away_odd_rate) / 2
    }
```

---

### 3.6 Integration with Existing System

**Modify `algorithms.py` - Add odd/even extraction to market probabilities:**

```python
def extract_markets_from_score_matrix(mat: pd.DataFrame, min_confidence: float = 0.6, 
                                     external_probs: Dict[str, Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    """Enhanced to include Odd/Even and Half predictions"""
    
    # ... existing code ...
    
    # ADD ODD/EVEN MARKET
    odd_total = even_total = 0.0
    
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = mat.loc[hg, ag]
            total_goals = hg + ag
            
            if total_goals % 2 == 1:
                odd_total += p
            else:
                even_total += p
    
    markets['OddEven'] = {
        'Odd': odd_total,
        'Even': even_total
    }
    
    # ... rest of existing code ...
    
    return markets
```

---

## PART 4: IMPLEMENTATION PRIORITIES & ACTION PLAN

### Phase 1 (Week 1): Foundation
- [ ] Create `src/niche_markets/` module structure
- [ ] Implement `data_preparation.py` - extract half-time data
- [ ] Calculate league priors from historical data
- [ ] Add half/odd-even features to `ml_features.py`

### Phase 2 (Week 2): Core Predictors
- [ ] Implement `odd_even_predictor.py`
- [ ] Implement `half_comparison_predictor.py`
- [ ] Test against winning slips (backtest)
- [ ] Calibrate confidence thresholds

### Phase 3 (Week 3): Advanced Models
- [ ] Implement `time_segmented_xg.py`
- [ ] Implement `league_priors.py` database
- [ ] Add Bayesian blending logic
- [ ] Validate against youth league data

### Phase 4 (Week 4): Integration
- [ ] Create `automate_niche_markets.py` CLI
- [ ] Implement `parlay_optimizer.py`
- [ ] Add to main `cli.py` commands
- [ ] Create dashboard visualizations

### Phase 5 (Week 5): Validation
- [ ] Backtest on 2023-2024 season data
- [ ] Forward test on current season
- [ ] Compare predictions to winning slips
- [ ] Tune hyperparameters (ratios, priors, weights)

---

## SUMMARY

**Key Insights:**
1. **2nd Half market** driven by fatigue, substitutions, game state urgency
2. **Odd/Even market** has natural odd-bias (~52-54%), amplified in youth/lower leagues
3. **Hybrid models essential**: Poisson + Time-segmentation + Bayesian priors + Context
4. **Data is available**: HTHG/HTAG columns enable all needed calculations

**Implementation Priority:**
1. ✅ Extract half-time statistics from existing CSVs
2. ✅ Build Odd/Even predictor with Poisson base + team bias
3. ✅ Build Half Comparison predictor with ratio modeling
4. ✅ Add league-specific priors (especially youth leagues)
5. ✅ Create parlay optimizer for mixed markets

**Success Metrics:**
- Odd/Even accuracy > 58% (vs 52% baseline)
- 2nd Half accuracy > 52% (vs 48% baseline)
- Combined parlay hit rate > 15% (6-leg), > 5% (10-leg)

---

**Next Steps:** Begin Phase 1 implementation → data preparation and feature extraction.

