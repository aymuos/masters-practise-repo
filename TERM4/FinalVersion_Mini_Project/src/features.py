import pandas as pd
import numpy as np
from scipy import signal
from collections import OrderedDict

def extract_features(df, debug=False, loop_id=None):
    """
    Extracts statistical and frequency-domain features from loop data.
    
    Args:
        df (pd.DataFrame): Input data with columns 'PV', 'SP', 'OP'.
        debug (bool): If True, saves a debug feature CSV.
        loop_id (str): Identifier for the loop (used in debug filename).
        
    Returns:
        dict: feature dictionary
    """
    features = {}
    
    # Pre-calculations
    pv = df['PV'].values
    sp = df['SP'].values
    op = df['OP'].values
    N = len(pv)

    # 1. Error calculation
    error_series = df['PV'] - df['SP']
    error = error_series.values
    
    # --- 1. Variance / Stability Features ---
    features['pv_std'] = np.std(pv)
    features['error_std'] = np.std(error)
    features['error_mean_abs'] = np.mean(np.abs(error))  # IAE-like
    
    # Drift (Slope of linear fit)
    # Normalized by signal range to make it scale-independent
    x = np.arange(N)
    pv_range = np.max(pv) - np.min(pv)
    if N > 1:
        slope, _, _, _, _ = stats_linregress(x, pv)
        if pv_range > 1e-9:
             features['pv_drift_slope'] = slope / pv_range
        else:
             features['pv_drift_slope'] = 0.0
    else:
        features['pv_drift_slope'] = 0.0

    # --- 2. Oscillation Features ---
    # a) Zero crossings of Mean-centered Error
    error_centered = error - np.mean(error)
    # Count sign changes
    zero_crossings = np.where(np.diff(np.signbit(error_centered)))[0]
    features['error_zero_crossings'] = len(zero_crossings) / N  # Normalized count
    
    # b) Spectral Analysis (PSD) on Error
    # Use Welch's method
    f, Pxx = signal.welch(error_centered, nperseg=min(256, N))
    if len(Pxx) > 0:
        idx_max = np.argmax(Pxx)
        dom_freq = f[idx_max]
        total_power = np.sum(Pxx)
        peak_power = Pxx[idx_max]
        
        features['psd_dom_freq'] = dom_freq
        features['psd_peak_ratio'] = peak_power / total_power if total_power > 0 else 0
    else:
        features['psd_dom_freq'] = 0
        features['psd_peak_ratio'] = 0

    # c) Autocorrelation (ACF) - detect periodicity
    n_lags = min(100, N // 4)
    if n_lags > 5:
        acf = np.correlate(error_centered, error_centered, mode='full')
        acf = acf[len(acf)//2:]  # Take positive lags
        # Normalize
        if acf[0] != 0:
            acf = acf / acf[0]
        
        # Find peaks in ACF (exclude lag 0)
        peaks, props = signal.find_peaks(acf[1:n_lags], height=0.2) 
        features['acf_num_peaks'] = len(peaks)
        if len(peaks) > 0:
            # Strength of first major peak
            features['acf_peak_val'] = props['peak_heights'][0]
            features['acf_peak_lag'] = peaks[0]
        else:
            features['acf_peak_val'] = 0
            features['acf_peak_lag'] = 0
            features['acf_peak_val'] = 0
            features['acf_peak_lag'] = 0
            
    # d) New ACF Features (Energy & Decay)
    if n_lags > 5 and len(acf) > 1:
        # Energy: Normalized sum of squares (Mean Square of ACF coefficients)
        acf_tails = acf[1:]
        features['err_acf_energy'] = np.sum(acf_tails**2) / len(acf_tails)
    else:
        features['err_acf_energy'] = 0.0

    # Decay Rate
    if len(peaks) >= 2:
        # Fit line to ln(peak_heights)
        # Model: ln(h) = -lambda * lag + C
        heights = props['peak_heights']
        if np.all(heights > 0):
            slope, _, _, _, _ = stats_linregress(peaks, np.log(heights))
            features['err_acf_decay_rate'] = -slope 
        else:
            features['err_acf_decay_rate'] = np.nan
    else:
        features['err_acf_decay_rate'] = np.nan

    # --- 3. Noise Features ---
    # High frequency volatility: Std of first difference
    d_pv = np.diff(pv)
    features['pv_diff_std'] = np.std(d_pv)
    
    # Spectral Entropy
    if len(Pxx) > 0:
        # Normalize PSD to get PMF
        psd_norm = Pxx / np.sum(Pxx)
        entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
        features['spectral_entropy'] = entropy
    else:
        features['spectral_entropy'] = 0

    # --- 4. Stiction Features ---
    # a) OP Flatness: Adaptive Percentile Threshold
    op_diff = np.abs(np.diff(op))
    # Adaptive threshold: 10th percentile of changes (captures 'near-zero' noise floor)
    eps = np.percentile(op_diff, 10) if len(op_diff) > 0 else 1e-6
    # Fallback if signal is perfectly constant (eps=0)
    if eps == 0: eps = 1e-6
        
    flat_mask = op_diff <= eps
    features['op_flatness_ratio'] = np.mean(flat_mask)
    
    # b) OP Jump Count (Step changes)
    # Threshold: 3 * std (robust to noise)
    jump_thresh = 3 * np.std(op_diff) if np.std(op_diff) > 1e-6 else 0.1
    jump_count = np.sum(op_diff > jump_thresh)
    features['op_jump_count'] = jump_count / len(op_diff) # Normalized by length
    
    # c) Cross-correlation Lag (OP -> PV)
    if N > 10:
        # Detrend and normalize
        op_norm = (op - np.mean(op))
        pv_norm = (pv - np.mean(pv))
        
        std_op = np.std(op_norm)
        std_pv = np.std(pv_norm)
        
        if std_op > 1e-9 and std_pv > 1e-9:
            cc = signal.correlate(pv_norm, op_norm, mode='full')
            lags = signal.correlation_lags(len(pv_norm), len(op_norm), mode='full')
            
            # Look for max correlation in positive lag (PV follows OP)
            mask = lags >= 0
            cc_pos = cc[mask]
            lags_pos = lags[mask]
            
            idx_max_cc = np.argmax(np.abs(cc_pos))
            # Normalize Correlation Coefficient (-1 to 1)
            # CC_norm = CC_raw / (N * std_x * std_y)
            features['cc_max_val'] = cc_pos[idx_max_cc] / (N * std_op * std_pv)
            features['cc_lag'] = lags_pos[idx_max_cc]
        else:
            features['cc_max_val'] = 0
            features['cc_lag'] = 0
    else:
        features['cc_max_val'] = 0
        features['cc_lag'] = 0

    # --- Debug: Dump Features ---
    if debug and loop_id:
        import os
        output_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        debug_dir = os.path.join(output_base, 'debugs', 'features')
        os.makedirs(debug_dir, exist_ok=True)
        debug_df = pd.DataFrame([features])
        debug_df.to_csv(os.path.join(debug_dir, f'features_loop_{loop_id}.csv'), index=False)

    return features

# Helper for drift (simple linear regression slope)

def stats_linregress(x, y):
    # Minimal implementation of linregress to avoid heavier imports if needed
    # slope = Cov(x,y) / Var(x)
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    cov = np.sum((x - mx) * (y - my))
    var_x = np.sum((x - mx)**2)
    if var_x == 0: return 0, 0, 0, 0, 0
    slope = cov / var_x
    return slope, 0, 0, 0, 0

if __name__ == "__main__":
    from utils_logger import setup_logging
    setup_logging("features")
    
    # Test with dummy data
    import numpy as np
    t = np.linspace(0, 100, 500)
    # Case 1: Oscillation
    df_osc = pd.DataFrame({
        'PV': np.sin(t),
        'SP': np.zeros(500),
        'OP': np.cos(t)
    })
    print("Oscillation Features:", extract_features(df_osc))
    
    # Case 2: Stiction (Square wave OP, Sawtooth PV usually, or Square PV)
    op_stiction = np.sign(np.sin(t)) # Square wave
    df_sti = pd.DataFrame({
        'PV': op_stiction * 0.8, # Tracking
        'SP': op_stiction,
        'OP': op_stiction
    })
    print("Stiction Features:", extract_features(df_sti))
