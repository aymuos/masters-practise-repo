import numpy as np
import pandas as pd
import json
import os

def norm_series(x, lo=5, hi=95):
    """
    Normalizes a pandas Series using percentiles.
    Returns (normalized_x, stats_dict)
    """
    p_lo = np.percentile(x, lo)
    p_hi = np.percentile(x, hi)
    denom = p_hi - p_lo + 1e-6
    
    x_norm = np.clip((x - p_lo) / denom, 0, 1)
    return x_norm, {'lo': p_lo, 'hi': p_hi, 'denom': denom}

def norm_scalar(val, stats):
    """
    Normalizes a scalar using pre-computed stats.
    """
    if stats is None: return 0.0
    return np.clip((val - stats['lo']) / stats['denom'], 0, 1)

def calc_severity_batch(df):
    """
    Calculates severity scores for the entire dataframe.
    Returns (df_with_severity, stats_dict)
    """
    df = df.copy()
    stats = {}
    
    # helper to get column safe
    def get_col(name):
        return df[name] if name in df.columns else pd.Series(np.zeros(len(df)))

    # 1. Normalize Components (Compute stats)
    # Oscillation Components
    acf, acf_stats = norm_series(get_col('acf_peak_val').fillna(0))
    psd, psd_stats = norm_series(get_col('psd_peak_ratio').fillna(0))
    ezc, ezc_stats = norm_series(get_col('error_zero_crossings').fillna(0))
    
    stats['acf_peak_val'] = acf_stats
    stats['psd_peak_ratio'] = psd_stats
    stats['error_zero_crossings'] = ezc_stats
    
    # Noise Components
    ent, ent_stats = norm_series(get_col('spectral_entropy').fillna(0))
    pvd, pvd_stats = norm_series(get_col('pv_diff_std').fillna(0))
    
    stats['spectral_entropy'] = ent_stats
    stats['pv_diff_std'] = pvd_stats
    
    # Disturbance Components
    err, err_stats = norm_series(get_col('error_std').fillna(0))
    stats['error_std'] = err_stats
    
    # --- FORMULAE ---
    
    # Oscillation: 0.45*ACF + 0.35*PSD + 0.20*ZC
    df['oscillation_severity'] = (0.45 * acf) + (0.35 * psd) + (0.20 * ezc)
    
    # Noise: 0.60*Entropy + 0.40*PVDiff
    df['noise_severity'] = (0.60 * ent) + (0.40 * pvd)
    
    # Disturbance: norm(error_std)
    df['disturbance_severity'] = err
    
    # Stiction
    if 'Stiction_Score' in df.columns:

        vals = df['Stiction_Score'].fillna(0)
        # Check max
        if vals.max() > 1.0:
            df['stiction_severity'] = np.clip(vals / 100.0, 0, 1)
        else:
            df['stiction_severity'] = np.clip(vals, 0, 1)
    else:
        df['stiction_severity'] = 0.0

    return df, stats

def calc_severity_single(features, stats):
    """
    Calculates severity for a single feature dict using stats.
    Returns dict of severities.
    """
    scores = {}
    
    # Helper
    def n(key):
        val = features.get(key, 0)
        if key not in stats: return 0.0
        return norm_scalar(val, stats[key])
        
    # Oscillation
    osc = (0.45 * n('acf_peak_val')) + (0.35 * n('psd_peak_ratio')) + (0.20 * n('error_zero_crossings'))
    scores['oscillation_severity'] = osc
    
    # Noise
    noi = (0.60 * n('spectral_entropy')) + (0.40 * n('pv_diff_std'))
    scores['noise_severity'] = noi
    
    # Disturbance
    scores['disturbance_severity'] = n('error_std')
    
    # Stiction
    scores['stiction_severity'] = 0.0
    
    return scores

def severity_band(score):
    if score is None: return "LOW"
    if score < 0.33: return "LOW"
    elif score < 0.66: return "MEDIUM"
    else: return "HIGH"
