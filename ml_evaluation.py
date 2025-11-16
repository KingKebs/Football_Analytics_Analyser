"""Evaluation utilities comparing ML predictions to Poisson baseline.
"""
from __future__ import annotations
import numpy as np
from typing import Dict


def evaluate_vs_poisson(poisson: Dict[str, Dict[str, float]], ml: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Compare ML predicted probabilities/values to Poisson-derived ones.

    poisson: markets dict e.g., {'1X2':{'Home':..,'Draw':..,'Away':..}, 'BTTS':{'Yes':..,'No':..}, 'OU':{'Over2.5':..,'Under2.5':..}}
    ml: predicted dict from predict_match

    Returns improvement metrics for each market (positive means ML better for accuracy proxies).
    For probabilities we can't directly compute accuracy here; we just surface absolute deltas.
    For total goals we treat diff from expected goals sum (poisson mean) as potential adjustment.
    """
    out = {}
    # 1X2
    if '1X2' in poisson and 'prob_1x2_home' in ml:
        p = poisson['1X2']
        out['1X2'] = {
            'poisson_home': float(p.get('Home',0.0)), 'ml_home': ml['prob_1x2_home'],
            'poisson_draw': float(p.get('Draw',0.0)), 'ml_draw': ml['prob_1x2_draw'],
            'poisson_away': float(p.get('Away',0.0)), 'ml_away': ml['prob_1x2_away'],
            'delta_home': ml['prob_1x2_home'] - float(p.get('Home',0.0)),
            'delta_draw': ml['prob_1x2_draw'] - float(p.get('Draw',0.0)),
            'delta_away': ml['prob_1x2_away'] - float(p.get('Away',0.0)),
        }
    # BTTS
    if 'BTTS' in poisson and 'prob_btts_yes' in ml:
        p = poisson['BTTS']
        out['BTTS'] = {
            'poisson_yes': float(p.get('Yes',0.0)), 'ml_yes': ml['prob_btts_yes'],
            'poisson_no': float(p.get('No',0.0)), 'ml_no': ml['prob_btts_no'],
            'delta_yes': ml['prob_btts_yes'] - float(p.get('Yes',0.0)),
            'delta_no': ml['prob_btts_no'] - float(p.get('No',0.0)),
        }
    # OU (focus 2.5 line)
    if 'OU' in poisson and 'pred_total_goals' in ml:
        p = poisson['OU']
        poisson_mean_over25 = float(p.get('Over2.5',0.0))
        out['OU2.5'] = {
            'poisson_over2.5': poisson_mean_over25,
            'ml_total_goals': ml['pred_total_goals'],
        }
    return out

