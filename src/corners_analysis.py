"""
Corners Analysis Script
=======================
Analyzes football match data to predict and understand corner generation patterns.

This script:
1. Loads CSV data with match statistics
2. Cleans and prepares features for analysis
3. Calculates total corners per match
4. Builds correlations between corners and match statistics
5. Estimates 1st vs 2nd half corner splits
6. Outputs model-ready features and trends
7. (Optional) Predicts corners for a specific home vs away matchup within a league

Usage:
  python corners_analysis.py                                # Analyze all leagues (default)
  python corners_analysis.py --league E0                   # Analyze specific league
  python corners_analysis.py --league E0,SP1               # Analyze multiple leagues
  python corners_analysis.py --league ALL                  # Explicitly analyze all leagues
  python corners_analysis.py --league E3 --home-team X --away-team Y  # Match-level prediction
  python corners_analysis.py --league E3 --home-only --top-n 10       # Show top 10 home corner teams
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import re
import math
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Optional XGBoost support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


# Supported league codes (matching download_all_tabs.py)
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


class CornersAnalyzer:
    """Analyzes corner patterns in football match data."""
    
    def __init__(self, csv_path, league_code=None):
        """
        Initialize the analyzer with a CSV file path.
        
        Args:
            csv_path (str): Path to the CSV file containing match data
            league_code (str, optional): League code for this analysis
        """
        self.csv_path = csv_path
        self.league_code = league_code
        self.df = None
        self.enriched_df = None
        self.correlations = None
        self.team_stats = None
        
    def load_data(self):
        """Load and validate CSV data."""
        try:
            self.df = pd.read_csv(self.csv_path)
            league_info = f" ({self.league_code})" if self.league_code else ""
            print(f"✓ Loaded {len(self.df)} matches from {self.csv_path}{league_info}")
            print(f"  Columns: {list(self.df.columns)}")
            return self.df
        except FileNotFoundError:
            print(f"✗ File not found: {self.csv_path}")
            return None
    
    def validate_corners_data(self):
        """Validate that corner columns exist and have data."""
        required_cols = ['HC', 'AC']
        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
            print(f"✗ Missing corner columns: {missing}")
            return False
        
        # Check for missing values
        hc_missing = self.df['HC'].isna().sum()
        ac_missing = self.df['AC'].isna().sum()
        
        if hc_missing > 0 or ac_missing > 0:
            print(f"⚠ Missing values - HC: {hc_missing}, AC: {ac_missing}")
            self.df['HC'].fillna(self.df['HC'].mean(), inplace=True)
            self.df['AC'].fillna(self.df['AC'].mean(), inplace=True)
        
        print(f"✓ Corners data validated")
        return True
    
    def clean_features(self):
        """Clean and prepare features for analysis."""
        # Define feature columns
        shot_features = ['HS', 'AS', 'HST', 'AST']
        foul_features = ['HF', 'AF']
        goal_features = ['HTHG', 'HTAG', 'FTHG', 'FTAG']
        
        all_feature_cols = shot_features + foul_features + goal_features
        
        # Fill missing values with column means
        for col in all_feature_cols:
            if col in self.df.columns:
                missing = self.df[col].isna().sum()
                if missing > 0:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
        
        print(f"✓ Features cleaned and prepared")
        return self.df
    
    def engineer_features(self):
        """Create new features for corner prediction."""
        df = self.df.copy()
        
        # Total corners
        df['Total_Corners'] = df['HC'] + df['AC']
        
        # Total shots and shots on target
        df['Total_Shots'] = df['HS'] + df['AS']
        df['Total_Shots_On_Target'] = df['HST'] + df['AST']
        
        # Total fouls
        df['Total_Fouls'] = df['HF'] + df['AF']
        
        # Total goals (full match)
        df['Total_Goals'] = df['FTHG'] + df['FTAG']
        
        # Halftime goals
        df['HT_Goals'] = df['HTHG'] + df['HTAG']
        
        # Second half goals (inferred)
        df['SH_Goals'] = df['Total_Goals'] - df['HT_Goals']
        
        # Goal dynamics
        df['Home_Goal_Diff'] = df['FTHG'] - df['FTAG']
        df['HT_Goal_Diff'] = df['HTHG'] - df['HTAG']
        
        # Match state indicators
        df['Home_Trailing_HT'] = (df['HTHG'] < df['HTAG']).astype(int)
        df['Away_Trailing_HT'] = (df['HTAG'] < df['HTHG']).astype(int)
        
        # Shots per foul ratio (match intensity)
        df['Shots_Per_Foul'] = df['Total_Shots'] / (df['Total_Fouls'] + 1)
        
        # Shot accuracy
        df['Shot_Accuracy'] = df['Total_Shots_On_Target'] / (df['Total_Shots'] + 1)
        
        # Cards (if available)
        if 'HY' in df.columns and 'AY' in df.columns:
            df['Total_Yellow_Cards'] = df['HY'] + df['AY']
        else:
            df['Total_Yellow_Cards'] = 0
        
        # --- Step 4: Interaction terms (non-linear relationships) ---
        try:
            df['HS_x_AS'] = df['HS'] * df['AS']
            df['HST_x_AST'] = df['HST'] * df['AST']
            df['Shots_x_Fouls'] = df['Total_Shots'] * df['Total_Fouls']
            df['HTDiff_x_Shots'] = df['HT_Goal_Diff'] * df['Total_Shots']
            df['HomeDiff_x_Fouls'] = df['Home_Goal_Diff'] * df['Total_Fouls']
        except Exception:
            # In case of missing base columns, fill zeros
            for col in ['HS_x_AS','HST_x_AST','Shots_x_Fouls','HTDiff_x_Shots','HomeDiff_x_Fouls']:
                if col not in df.columns:
                    df[col] = 0.0

        self.enriched_df = df
        print(f"✓ {len(df.columns) - len(self.df.columns)} new features engineered")
        return df
    
    def calculate_correlations(self):
        """Calculate correlations between corners and other match statistics."""
        df = self.enriched_df
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation with total corners
        correlations = df[numeric_cols].corr()['Total_Corners'].sort_values(ascending=False)
        
        self.correlations = correlations
        
        print(f"\n{'='*60}")
        print(f"CORNER CORRELATIONS (Top 15)")
        print(f"{'='*60}")
        for feature, corr in correlations.head(15).items():
            if feature != 'Total_Corners':
                print(f"{feature:30s}: {corr:7.4f}")
        
        return correlations
    
    def estimate_half_split(self):
        """
        Estimate 1st half vs 2nd half corners using multiple methods.
        
        Methods:
        1. Historical baseline (40% 1st half, 60% 2nd half)
        2. Goal timing adjustment (early goals shift to 1st half)
        3. Match intensity adjustment (high fouls → more 2nd half corners)
        """
        df = self.enriched_df.copy()
        
        # Method 1: Baseline split (40/60)
        baseline_1h_ratio = 0.40
        
        # Method 2: Adjust based on halftime goals
        # Early goals (more goals in 1st half) → more 1st half corners
        goal_adjustment = (df['HT_Goals'] / (df['Total_Goals'] + 1)) * 0.15
        
        # Method 3: Adjust based on match intensity
        # High fouls in 2nd half indicate more defensive play → more corners
        intensity_adjustment = -0.05  # slight shift toward 2nd half corners
        
        # Combined 1st half ratio
        df['Est_1H_Corner_Ratio'] = baseline_1h_ratio + goal_adjustment + intensity_adjustment
        
        # Clamp between 0.25 and 0.55 (reasonable bounds)
        df['Est_1H_Corner_Ratio'] = df['Est_1H_Corner_Ratio'].clip(0.25, 0.55)
        
        # Calculate estimated corners
        df['Est_1H_Corners'] = (df['Total_Corners'] * df['Est_1H_Corner_Ratio']).round(1)
        df['Est_2H_Corners'] = (df['Total_Corners'] * (1 - df['Est_1H_Corner_Ratio'])).round(1)
        
        self.enriched_df = df
        
        print(f"\n{'='*60}")
        print(f"HALF-SPLIT ESTIMATION SAMPLE (First 5 matches)")
        print(f"{'='*60}")
        sample_cols = ['Total_Corners', 'Est_1H_Corner_Ratio', 'Est_1H_Corners', 'Est_2H_Corners']
        print(df[sample_cols].head().to_string())
        
        return df
    
    def calculate_team_stats(self):
        """Calculate cumulative team statistics for corner prediction."""
        df = self.enriched_df
        
        # Separate home and away perspective
        home_stats = df.groupby('HomeTeam').agg({
            'HC': ['mean', 'std', 'count'],
            'AC': 'mean',
            'Total_Corners': 'mean',
            'HS': 'mean',
            'HF': 'mean',
            'Total_Fouls': 'mean',
        }).round(2)
        
        away_stats = df.groupby('AwayTeam').agg({
            'AC': ['mean', 'std', 'count'],
            'HC': 'mean',
            'Total_Corners': 'mean',
            'AS': 'mean',
            'AF': 'mean',
            'Total_Fouls': 'mean',
        }).round(2)
        
        home_stats.columns = ['Avg_Corners_For', 'Std_Corners_For', 'Matches', 
                              'Avg_Corners_Against', 'Avg_Total_Corners', 
                              'Avg_Shots', 'Avg_Fouls', 'Avg_Total_Fouls']
        
        away_stats.columns = ['Avg_Corners_For', 'Std_Corners_For', 'Matches', 
                              'Avg_Corners_Against', 'Avg_Total_Corners', 
                              'Avg_Shots', 'Avg_Fouls', 'Avg_Total_Fouls']
        
        self.team_stats = (home_stats, away_stats)
        
        print(f"\n{'='*60}")
        print(f"TEAM STATISTICS (Top 5 by Avg Corners)")
        print(f"{'='*60}")
        print("\nHome Teams:")
        print(home_stats.sort_values('Avg_Corners_For', ascending=False).head())
        
        return home_stats, away_stats

    # New helper methods
    def display_top_teams(self, top_n=5, home_only=False, away_only=False):
        """Display top N teams by average corners for home or away context."""
        if not self.team_stats:
            print("Team stats not computed yet.")
            return
        home_stats, away_stats = self.team_stats
        if home_only:
            print(f"\nTop {top_n} Home Teams by Avg Corners For")
            print(home_stats.sort_values('Avg_Corners_For', ascending=False).head(top_n)[['Avg_Corners_For','Std_Corners_For','Matches']])
        elif away_only:
            print(f"\nTop {top_n} Away Teams by Avg Corners For")
            print(away_stats.sort_values('Avg_Corners_For', ascending=False).head(top_n)[['Avg_Corners_For','Std_Corners_For','Matches']])
        else:
            print(f"\nTop {top_n} Home Teams (Avg Corners For)")
            print(home_stats.sort_values('Avg_Corners_For', ascending=False).head(top_n)[['Avg_Corners_For','Std_Corners_For','Matches']])
            print(f"\nTop {top_n} Away Teams (Avg Corners For)")
            print(away_stats.sort_values('Avg_Corners_For', ascending=False).head(top_n)[['Avg_Corners_For','Std_Corners_For','Matches']])

    def find_team(self, query, home=True):
        """Find a team name (exact or fuzzy) among home or away team stats."""
        if not self.team_stats:
            return None
        home_stats, away_stats = self.team_stats
        stats_df = home_stats if home else away_stats
        # Exact
        if query in stats_df.index:
            return query
        # Case-insensitive exact
        for name in stats_df.index:
            if name.lower() == query.lower():
                return name
        # Fuzzy contains
        for name in stats_df.index:
            if query.lower() in name.lower():
                return name
        return None

    def _compute_recency_weights(self, df: pd.DataFrame, half_life_days: int = 180) -> np.ndarray:
        """Compute recency weights from Date column using exponential decay.
        If Date missing/unparseable, return uniform weights of 1.0.
        """
        if 'Date' not in df.columns:
            return np.ones(len(df), dtype=float)
        try:
            dates = pd.to_datetime(df['Date'], errors='coerce')
            max_date = dates.max()
            if pd.isna(max_date):
                return np.ones(len(df), dtype=float)
            age_days = (max_date - dates).dt.days
            age_days = age_days.fillna(0)
            lam = np.log(2) / max(half_life_days, 1)
            w = np.exp(-lam * age_days.astype(float))
            # Normalize weights for stability
            mean_w = w.mean()
            if mean_w and mean_w > 0:
                w = w / mean_w
            return w.values
        except Exception:
            return np.ones(len(df), dtype=float)

    def train_models(self):
        """Train simple regression models to predict Total_Corners and 1H ratio.
        Saves coefficients as JSON. Also evaluates non-linear ensembles via CV.
        """
        if not SKLEARN_AVAILABLE:
            print("⚠ scikit-learn not available. Install it to use --train-model.")
            return None
        if self.enriched_df is None:
            print("Run feature engineering before training.")
            return None
        df = self.enriched_df.copy()
        target_total = df['Total_Corners']
        target_ratio = df['Est_1H_Corner_Ratio'] if 'Est_1H_Corner_Ratio' in df.columns else None
        feature_cols = [c for c in ['HS','AS','HST','AST','HF','AF','FTHG','FTAG','HTHG','HTAG','Total_Shots','Total_Shots_On_Target','Shots_Per_Foul','Shot_Accuracy',
                                    'HS_x_AS','HST_x_AST','Shots_x_Fouls','HTDiff_x_Shots','HomeDiff_x_Fouls'] if c in df.columns]
        X = df[feature_cols].values
        w = self._compute_recency_weights(df)
        # In-sample linear fit (coeffs)
        model_total = LinearRegression().fit(X, target_total)
        metrics = {
            'total_corners_r2': round(model_total.score(X, target_total),4),
            'total_corners_mae': round(float(np.mean(np.abs(model_total.predict(X)-target_total))),3),
            'features_used': feature_cols,
            'coefficients': {f: round(float(c),4) for f,c in zip(feature_cols, model_total.coef_)},
            'intercept': round(float(model_total.intercept_),4)
        }
        # In-sample ratio linear model (for visibility/comparison)
        model_ratio = None
        if target_ratio is not None:
            model_ratio = LinearRegression().fit(X, target_ratio)
            metrics.update({
                'ratio_r2': round(float(model_ratio.score(X, target_ratio)),4),
                'ratio_mae': round(float(np.mean(np.abs(model_ratio.predict(X)-target_ratio))),3),
                'ratio_intercept': round(float(model_ratio.intercept_),4),
                'ratio_coefficients': {f: round(float(c),4) for f,c in zip(feature_cols, model_ratio.coef_)}
            })
        # Cross-validation (unweighted) for Linear
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2_total = cross_val_score(LinearRegression(), X, target_total, cv=kf, scoring='r2')
        cv_mae_total = -cross_val_score(LinearRegression(), X, target_total, cv=kf, scoring='neg_mean_absolute_error')
        metrics.update({
            'cv_total_corners_r2_mean': round(float(np.mean(cv_r2_total)),4),
            'cv_total_corners_r2_std': round(float(np.std(cv_r2_total)),4),
            'cv_total_corners_mae_mean': round(float(np.mean(cv_mae_total)),3),
            'cv_total_corners_mae_std': round(float(np.std(cv_mae_total)),3),
        })
        if target_ratio is not None:
            cv_r2_ratio = cross_val_score(LinearRegression(), X, target_ratio, cv=kf, scoring='r2')
            cv_mae_ratio = -cross_val_score(LinearRegression(), X, target_ratio, cv=kf, scoring='neg_mean_absolute_error')
            metrics.update({
                'cv_ratio_r2_mean': round(float(np.mean(cv_r2_ratio)),4),
                'cv_ratio_r2_std': round(float(np.std(cv_r2_ratio)),4),
                'cv_ratio_mae_mean': round(float(np.mean(cv_mae_ratio)),3),
                'cv_ratio_mae_std': round(float(np.std(cv_mae_ratio)),3),
            })
        # Unweighted RF CV
        rf_params = dict(n_estimators=300, random_state=42, n_jobs=-1)
        rf_total = RandomForestRegressor(**rf_params)
        rf_r2_total = cross_val_score(rf_total, X, target_total, cv=kf, scoring='r2')
        rf_mae_total = -cross_val_score(rf_total, X, target_total, cv=kf, scoring='neg_mean_absolute_error')
        metrics.update({
            'rf_total_r2_cv_mean': round(float(np.mean(rf_r2_total)),4),
            'rf_total_r2_cv_std': round(float(np.std(rf_r2_total)),4),
            'rf_total_mae_cv_mean': round(float(np.mean(rf_mae_total)),3),
            'rf_total_mae_cv_std': round(float(np.std(rf_mae_total)),3),
        })
        if target_ratio is not None:
            rf_ratio = RandomForestRegressor(**rf_params)
            rf_r2_ratio = cross_val_score(rf_ratio, X, target_ratio, cv=kf, scoring='r2')
            rf_mae_ratio = -cross_val_score(rf_ratio, X, target_ratio, cv=kf, scoring='neg_mean_absolute_error')
            metrics.update({
                'rf_ratio_r2_cv_mean': round(float(np.mean(rf_r2_ratio)),4),
                'rf_ratio_r2_cv_std': round(float(np.std(rf_r2_ratio)),4),
                'rf_ratio_mae_cv_mean': round(float(np.mean(rf_mae_ratio)),3),
                'rf_ratio_mae_cv_std': round(float(np.std(rf_mae_ratio)),3),
            })
        # Weighted CV (manual) Linear & RF
        lin = LinearRegression()
        wr2, wmae = [], []
        for tr, te in kf.split(X):
            lin.fit(X[tr], target_total.iloc[tr], sample_weight=w[tr])
            y_pred = lin.predict(X[te])
            y_true = target_total.iloc[te].values
            wt = w[te]
            y_bar = np.average(target_total.iloc[tr].values, weights=w[tr])
            ss_res = np.average((y_true - y_pred)**2, weights=wt)
            ss_tot = np.average((y_true - y_bar)**2, weights=wt)
            wr2.append(1 - ss_res/ss_tot if ss_tot > 1e-8 else 0.0)
            wmae.append(np.average(np.abs(y_true - y_pred), weights=wt))
        metrics.update({
            'wcv_total_linear_r2_mean': round(float(np.mean(wr2)),4),
            'wcv_total_linear_r2_std': round(float(np.std(wr2)),4),
            'wcv_total_linear_mae_mean': round(float(np.mean(wmae)),3),
            'wcv_total_linear_mae_std': round(float(np.std(wmae)),3),
        })
        wrf_r2, wrf_mae = [], []
        for tr, te in kf.split(X):
            rf = RandomForestRegressor(**rf_params)
            rf.fit(X[tr], target_total.iloc[tr], sample_weight=w[tr])
            y_pred = rf.predict(X[te])
            y_true = target_total.iloc[te].values
            wt = w[te]
            y_bar = np.average(target_total.iloc[tr].values, weights=w[tr])
            ss_res = np.average((y_true - y_pred)**2, weights=wt)
            ss_tot = np.average((y_true - y_bar)**2, weights=wt)
            wrf_r2.append(1 - ss_res/ss_tot if ss_tot > 1e-8 else 0.0)
            wrf_mae.append(np.average(np.abs(y_true - y_pred), weights=wt))
        metrics.update({
            'wcv_total_rf_r2_mean': round(float(np.mean(wrf_r2)),4),
            'wcv_total_rf_r2_std': round(float(np.std(wrf_r2)),4),
            'wcv_total_rf_mae_mean': round(float(np.mean(wrf_mae)),3),
            'wcv_total_rf_mae_std': round(float(np.std(wrf_mae)),3),
        })
        if target_ratio is not None:
            wr2, wmae = [], []
            for tr, te in kf.split(X):
                lin = LinearRegression()
                lin.fit(X[tr], target_ratio.iloc[tr], sample_weight=w[tr])
                y_pred = lin.predict(X[te])
                y_true = target_ratio.iloc[te].values
                wt = w[te]
                y_bar = np.average(target_ratio.iloc[tr].values, weights=w[tr])
                ss_res = np.average((y_true - y_pred)**2, weights=wt)
                ss_tot = np.average((y_true - y_bar)**2, weights=wt)
                wr2.append(1 - ss_res/ss_tot if ss_tot > 1e-8 else 0.0)
                wmae.append(np.average(np.abs(y_true - y_pred), weights=wt))
            metrics.update({
                'wcv_ratio_linear_r2_mean': round(float(np.mean(wr2)),4),
                'wcv_ratio_linear_r2_std': round(float(np.std(wr2)),4),
                'wcv_ratio_linear_mae_mean': round(float(np.mean(wmae)),3),
                'wcv_ratio_linear_mae_std': round(float(np.std(wmae)),3),
            })
            wr2, wmae = [], []
            for tr, te in kf.split(X):
                rf = RandomForestRegressor(**rf_params)
                rf.fit(X[tr], target_ratio.iloc[tr], sample_weight=w[tr])
                y_pred = rf.predict(X[te])
                y_true = target_ratio.iloc[te].values
                wt = w[te]
                y_bar = np.average(target_ratio.iloc[tr].values, weights=w[tr])
                ss_res = np.average((y_true - y_pred)**2, weights=wt)
                ss_tot = np.average((y_true - y_bar)**2, weights=wt)
                wr2.append(1 - ss_res/ss_tot if ss_tot > 1e-8 else 0.0)
                wmae.append(np.average(np.abs(y_true - y_pred), weights=wt))
            metrics.update({
                'wcv_ratio_rf_r2_mean': round(float(np.mean(wr2)),4),
                'wcv_ratio_rf_r2_std': round(float(np.std(wr2)),4),
                'wcv_ratio_rf_mae_mean': round(float(np.mean(wmae)),3),
                'wcv_ratio_rf_mae_std': round(float(np.std(wmae)),3),
            })
        # Optional XGBoost (unweighted CV)
        if 'XGBOOST_AVAILABLE' in globals() and XGBOOST_AVAILABLE:
            xgb_total = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method='hist')
            xgb_r2_total = cross_val_score(xgb_total, X, target_total, cv=kf, scoring='r2')
            xgb_mae_total = -cross_val_score(xgb_total, X, target_total, cv=kf, scoring='neg_mean_absolute_error')
            metrics.update({
                'xgb_total_r2_cv_mean': round(float(np.mean(xgb_r2_total)),4),
                'xgb_total_r2_cv_std': round(float(np.std(xgb_r2_total)),4),
                'xgb_total_mae_cv_mean': round(float(np.mean(xgb_mae_total)),3),
                'xgb_total_mae_cv_std': round(float(np.std(xgb_mae_total)),3),
            })
            if target_ratio is not None:
                xgb_ratio = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method='hist')
                xgb_r2_ratio = cross_val_score(xgb_ratio, X, target_ratio, cv=kf, scoring='r2')
                xgb_mae_ratio = -cross_val_score(xgb_ratio, X, target_ratio, cv=kf, scoring='neg_mean_absolute_error')
                metrics.update({
                    'xgb_ratio_r2_cv_mean': round(float(np.mean(xgb_r2_ratio)),4),
                    'xgb_ratio_r2_cv_std': round(float(np.std(xgb_r2_ratio)),4),
                    'xgb_ratio_mae_cv_mean': round(float(np.mean(xgb_mae_ratio)),3),
                    'xgb_ratio_mae_cv_std': round(float(np.std(xgb_mae_ratio)),3),
                })
        # Export metrics only (model persistence could be added later)
        corners_dir = os.path.join('data','corners')
        os.makedirs(corners_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(corners_dir, f'model_metrics_{self.league_code}_{ts}.json')
        with open(out_path,'w') as f:
            json.dump(metrics, f, indent=2)

        # Print model metrics for visibility
        print(f"\n{'='*60}")
        print("REGRESSION MODEL METRICS")
        print(f"{'='*60}")
        print("Total Corners Model (Linear, in-sample):")
        print(f"  R² Score:  {metrics.get('total_corners_r2', 'N/A')}")
        print(f"  MAE:       {metrics.get('total_corners_mae', 'N/A')}")
        print("Total Corners Model (Linear, 5-fold CV):")
        print(f"  R² mean±std:  {metrics.get('cv_total_corners_r2_mean','N/A')} ± {metrics.get('cv_total_corners_r2_std','N/A')}")
        print(f"  MAE mean±std: {metrics.get('cv_total_corners_mae_mean','N/A')} ± {metrics.get('cv_total_corners_mae_std','N/A')}")
        print("Total Corners Model (RandomForest, 5-fold CV):")
        print(f"  R² mean±std:  {metrics.get('rf_total_r2_cv_mean','N/A')} ± {metrics.get('rf_total_r2_cv_std','N/A')}")
        print(f"  MAE mean±std: {metrics.get('rf_total_mae_cv_mean','N/A')} ± {metrics.get('rf_total_mae_cv_std','N/A')}")
        if 'xgb_total_r2_cv_mean' in metrics:
            print("Total Corners Model (XGBoost, 5-fold CV):")
            print(f"  R² mean±std:  {metrics.get('xgb_total_r2_cv_mean','N/A')} ± {metrics.get('xgb_total_r2_cv_std','N/A')}")
            print(f"  MAE mean±std: {metrics.get('xgb_total_mae_cv_mean','N/A')} ± {metrics.get('xgb_total_mae_cv_std','N/A')}")
        print("Total Corners Model (Weighted CV):")
        print(f"  Linear R² mean±std: {metrics.get('wcv_total_linear_r2_mean','N/A')} ± {metrics.get('wcv_total_linear_r2_std','N/A')}")
        print(f"  Linear MAE mean±std: {metrics.get('wcv_total_linear_mae_mean','N/A')} ± {metrics.get('wcv_total_linear_mae_std','N/A')}")
        print(f"  RF R² mean±std:      {metrics.get('wcv_total_rf_r2_mean','N/A')} ± {metrics.get('wcv_total_rf_r2_std','N/A')}")
        print(f"  RF MAE mean±std:     {metrics.get('wcv_total_rf_mae_mean','N/A')} ± {metrics.get('wcv_total_rf_mae_std','N/A')}")
        if target_ratio is not None:
            print(f"\nHalf-Split Ratio Model (Linear, in-sample):")
            print(f"  R² Score:  {metrics.get('ratio_r2', 'N/A')}")
            print(f"  MAE:       {metrics.get('ratio_mae', 'N/A')}")
            print("Half-Split Ratio Model (Linear, 5-fold CV):")
            print(f"  R² mean±std:  {metrics.get('cv_ratio_r2_mean','N/A')} ± {metrics.get('cv_ratio_r2_std','N/A')}")
            print(f"  MAE mean±std: {metrics.get('cv_ratio_mae_mean','N/A')} ± {metrics.get('cv_ratio_mae_std','N/A')}")
            print("Half-Split Ratio Model (RandomForest, 5-fold CV):")
            print(f"  R² mean±std:  {metrics.get('rf_ratio_r2_cv_mean','N/A')} ± {metrics.get('rf_ratio_r2_cv_std','N/A')}")
            print(f"  MAE mean±std: {metrics.get('rf_ratio_mae_cv_mean','N/A')} ± {metrics.get('rf_ratio_mae_cv_std','N/A')}")
            if 'xgb_ratio_r2_cv_mean' in metrics:
                print("Half-Split Ratio Model (XGBoost, 5-fold CV):")
                print(f"  R² mean±std:  {metrics.get('xgb_ratio_r2_cv_mean','N/A')} ± {metrics.get('xgb_ratio_r2_cv_std','N/A')}")
                print(f"  MAE mean±std: {metrics.get('xgb_ratio_mae_cv_mean','N/A')} ± {metrics.get('xgb_ratio_mae_cv_std','N/A')}")
            print("Half-Split Ratio Model (Weighted CV):")
            print(f"  Linear R² mean±std: {metrics.get('wcv_ratio_linear_r2_mean','N/A')} ± {metrics.get('wcv_ratio_linear_r2_std','N/A')}")
            print(f"  Linear MAE mean±std: {metrics.get('wcv_ratio_linear_mae_mean','N/A')} ± {metrics.get('wcv_ratio_linear_mae_std','N/A')}")
            print(f"  RF R² mean±std:      {metrics.get('wcv_ratio_rf_r2_mean','N/A')} ± {metrics.get('wcv_ratio_rf_r2_std','N/A')}")
            print(f"  RF MAE mean±std:     {metrics.get('wcv_ratio_rf_mae_mean','N/A')} ± {metrics.get('wcv_ratio_rf_mae_std','N/A')}")
        # ...existing trailing prints...
        return metrics

    # --- Added: Match-level prediction using team stats ---
    def predict_match_corners(self, home_team: str, away_team: str):
        """Predict expected corners (home, away, total, 1H/2H split) for a matchup using team stats.
        Method:
          - Expected home corners = avg(home home-corners for) + avg(away conceded corners against) / 2
          - Expected away corners = avg(away away-corners for) + avg(home conceded corners against) / 2
          - Total = sum; 1H split uses median Est_1H_Corner_Ratio or baseline.
        Returns dict with predictions.
        """
        if not self.team_stats:
            raise ValueError("Team stats not computed; run calculate_team_stats() first.")
        home_stats, away_stats = self.team_stats
        if home_team not in home_stats.index:
            raise ValueError(f"Home team '{home_team}' not found in stats")
        if away_team not in away_stats.index:
            raise ValueError(f"Away team '{away_team}' not found in stats")
        # Extract stats
        h_row = home_stats.loc[home_team]
        a_row = away_stats.loc[away_team]
        # Conceded approximations (home: AC mean; away: HC mean in away_stats)
        home_for = float(h_row['Avg_Corners_For']); home_conc_by_away = float(a_row['Avg_Corners_Against'])
        away_for = float(a_row['Avg_Corners_For']); away_conc_by_home = float(h_row['Avg_Corners_Against'])
        exp_home = (home_for + away_conc_by_home) / 2.0
        exp_away = (away_for + home_conc_by_away) / 2.0
        total = exp_home + exp_away
        ratio_series = self.enriched_df['Est_1H_Corner_Ratio'] if 'Est_1H_Corner_Ratio' in self.enriched_df.columns else pd.Series([0.40])
        ratio = float(ratio_series.median()) if not ratio_series.empty else 0.40
        one_h = total * ratio
        two_h = total - one_h
        return {
            'home_team': home_team,
            'away_team': away_team,
            'expected_home_corners': round(exp_home, 2),
            'expected_away_corners': round(exp_away, 2),
            'expected_total_corners': round(total, 2),
            'est_1h_ratio': round(ratio, 3),
            'expected_1h_corners': round(one_h, 2),
            'expected_2h_corners': round(two_h, 2)
        }

# --- Added: Utility functions for multi-league processing & CLI ---

def find_league_csv(league_code: str) -> str:
    paths = [
        os.path.join('football-data', 'all-euro-football', f'{league_code}.csv'),
        os.path.join('football-data', f'{league_code}.csv'),
        f'{league_code}.csv'
    ]
    for p in paths:
        if os.path.isfile(p):
            return p
    return ''


def process_league(league_code: str, args) -> dict:
    csv_path = find_league_csv(league_code) or args.file
    if not csv_path or not os.path.isfile(csv_path):
        print(f"Skipping {league_code}: CSV not found")
        return {'league': league_code, 'status': 'missing'}
    analyzer = CornersAnalyzer(csv_path, league_code=league_code)
    if analyzer.load_data() is None:
        return {'league': league_code, 'status': 'load_failed'}
    if not analyzer.validate_corners_data():
        return {'league': league_code, 'status': 'invalid_corners'}
    analyzer.clean_features()
    analyzer.engineer_features()
    analyzer.estimate_half_split()
    analyzer.calculate_correlations()
    analyzer.calculate_team_stats()
    metrics = None
    if args.train_model:
        metrics = analyzer.train_models()
    # Top team views
    if args.top_n > 0:
        analyzer.display_top_teams(top_n=args.top_n, home_only=args.home_only, away_only=args.away_only)
    match_pred = None
    if args.home_team and args.away_team:
        try:
            match_pred = analyzer.predict_match_corners(args.home_team, args.away_team)
            print(f"\nMATCH PREDICTION: {args.home_team} vs {args.away_team}")
            print(json.dumps(match_pred, indent=2))
        except Exception as e:
            print(f"Match prediction failed: {e}")
    # Save enriched features if requested
    if args.save_enriched:
        out_dir = args.output_dir or 'data/corners'
        os.makedirs(out_dir, exist_ok=True)
        enriched_path = os.path.join(out_dir, f'enriched_{league_code}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        analyzer.enriched_df.to_csv(enriched_path, index=False)
        print(f"Enriched feature set saved: {enriched_path}")
    return {
        'league': league_code,
        'rows': len(analyzer.df) if analyzer.df is not None else 0,
        'match_prediction': match_pred,
        'model_metrics': metrics,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Corner Analysis CLI')
    p.add_argument('--league', help='League code(s) comma-separated or ALL', default='ALL')
    p.add_argument('--file', help='Override CSV path (single league)')
    p.add_argument('--home-team', help='Home team for match-level prediction')
    p.add_argument('--away-team', help='Away team for match-level prediction')
    p.add_argument('--top-n', type=int, default=0, help='Display top N teams by corners (0=skip)')
    p.add_argument('--home-only', action='store_true', help='Show only home team stats')
    p.add_argument('--away-only', action='store_true', help='Show only away team stats')
    p.add_argument('--train-model', action='store_true', help='Train regression models for corners')
    p.add_argument('--save-enriched', action='store_true', help='Save enriched engineered feature CSV')
    p.add_argument('--output-dir', help='Directory for outputs (features, metrics)', default='data/corners')
    p.add_argument('--json-summary', action='store_true', help='Print JSON summary at end')
    p.add_argument('--use-parsed-all', action='store_true', help='Generate corner predictions for all parsed fixtures (todays_fixtures_<date>)')
    p.add_argument('--fixtures-date', help='Date (YYYYMMDD) of parsed fixtures file; default today', default=None)
    p.add_argument('--min-team-matches', type=int, default=5, help='Minimum matches for a team to allow prediction (below -> skip)')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return p.parse_args()

def _load_parsed_fixtures(date_str: str = None, data_dir: str = 'data') -> pd.DataFrame:
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')
    csv_path = os.path.join(data_dir, f'todays_fixtures_{date_str}.csv')
    json_path = os.path.join(data_dir, f'todays_fixtures_{date_str}.json')
    path = None
    if os.path.exists(csv_path):
        path = csv_path
    elif os.path.exists(json_path):
        path = json_path
    else:
        # fallback newest
        candidates = sorted([p for p in Path(data_dir).glob('todays_fixtures_*.json')] + [p for p in Path(data_dir).glob('todays_fixtures_*.csv')], key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            path = str(candidates[0])
    if not path:
        return pd.DataFrame()
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_json(path)
    except Exception:
        return pd.DataFrame()
    # Normalize columns to expected names
    cols = {c.lower(): c for c in df.columns}
    ren = {}
    if 'home' in cols: ren[cols['home']] = 'HomeTeam'
    if 'away' in cols: ren[cols['away']] = 'AwayTeam'
    if 'league' in cols: ren[cols['league']] = 'League'
    if 'competition' in cols and 'League' not in ren: ren[cols['competition']] = 'Competition'
    df = df.rename(columns=ren)
    for col in ['HomeTeam','AwayTeam','League','Competition']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def main():
    args = parse_args()
    np.random.seed(args.seed)
    if args.home_only and args.away_only:
        print("Cannot set both --home-only and --away-only; choose one.")
        return 1
    if args.league.upper() == 'ALL':
        leagues = SUPPORTED_LEAGUES
    else:
        leagues = [l.strip().upper() for l in args.league.split(',') if l.strip()]
    summaries = []
    analyzers = {}
    for lg in leagues:
        summary = process_league(lg, args)
        summaries.append(summary)
        # Retain analyzer for fixture predictions if enabled
        # We refit lightweight analyzer to get team_stats reference
        if args.use_parsed_all and summary.get('status') not in ('missing','load_failed','invalid_corners'):
            csv_path = find_league_csv(lg) or args.file
            if csv_path and os.path.isfile(csv_path):
                analyzer = CornersAnalyzer(csv_path, league_code=lg)
                if analyzer.load_data() is not None and analyzer.validate_corners_data():
                    analyzer.clean_features(); analyzer.engineer_features(); analyzer.estimate_half_split(); analyzer.calculate_team_stats()
                    analyzers[lg] = analyzer
    if args.json_summary:
        print("\n=== JSON SUMMARY ===")
        print(json.dumps(summaries, indent=2))
    # Parsed fixture corner predictions
    if args.use_parsed_all:
        date_str = args.fixtures_date or datetime.now().strftime('%Y%m%d')
        fixtures_df = _load_parsed_fixtures(date_str)
        if fixtures_df.empty:
            print(f"No parsed fixtures found for date {date_str}")
        else:
            print(f"\nParsed fixtures loaded ({len(fixtures_df)}) for date {date_str}")
            predictions = []
            skips = []
            for _, row in fixtures_df.iterrows():
                home = row.get('HomeTeam') or row.get('home')
                away = row.get('AwayTeam') or row.get('away')
                league = str(row.get('League') or '').upper()
                if not home or not away:
                    skips.append({'home': home, 'away': away, 'reason': 'missing_team'})
                    continue
                if league and league not in analyzers:
                    # Skip leagues not processed
                    skips.append({'home': home, 'away': away, 'league': league, 'reason': 'league_not_processed'})
                    continue
                # If league absent attempt heuristic: choose league with both teams present
                analyzer = analyzers.get(league)
                if not analyzer:
                    # try find league by team presence
                    for lg, an in analyzers.items():
                        hs, as_ = an.team_stats
                        if home in hs.index and away in as_.index:
                            analyzer = an; league = lg; break
                if not analyzer:
                    skips.append({'home': home, 'away': away, 'league': league, 'reason': 'no_inference'})
                    continue
                # Team occurrence threshold
                hs, as_ = analyzer.team_stats
                h_matches = int(hs.loc[home]['Matches']) if home in hs.index else 0
                a_matches = int(as_.loc[away]['Matches']) if away in as_.index else 0
                if h_matches < args.min_team_matches or a_matches < args.min_team_matches:
                    skips.append({'home': home, 'away': away, 'league': league, 'reason': 'insufficient_history', 'home_matches': h_matches, 'away_matches': a_matches})
                    continue
                try:
                    pred = analyzer.predict_match_corners(home, away)
                    pred['league_code'] = league
                    pred['source_file'] = row.get('file', f'todays_fixtures_{date_str}')
                    predictions.append(pred)
                except Exception as e:
                    skips.append({'home': home, 'away': away, 'league': league, 'reason': 'prediction_error', 'error': str(e)})
            # Save predictions
            out_dir = args.output_dir or 'data/corners'
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'parsed_corners_predictions_{date_str}.json')
            with open(out_path, 'w') as f:
                json.dump({'date': date_str, 'predictions': predictions, 'skipped': skips}, f, indent=2)
            print(f"\nParsed fixture corner predictions saved: {out_path}")
            print(f"Predictions: {len(predictions)} | Skipped: {len(skips)}")
            if skips:
                reason_counts = {}
                for s in skips:
                    r = s.get('reason','unknown'); reason_counts[r] = reason_counts.get(r,0)+1
                print("Skip reasons: " + ', '.join(f"{k}={v}" for k,v in reason_counts.items()))
    return 0

if __name__ == '__main__':
    main()
