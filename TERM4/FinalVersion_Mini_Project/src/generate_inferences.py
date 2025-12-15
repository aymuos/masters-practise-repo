import pandas as pd
import numpy as np
from scipy import signal, stats
import os
import glob
from src import config
from src.utils_logger import setup_logger

logger = setup_logger("gen_inferences", log_file=config.LOGS_DIR / "gen_inferences.log")


# 1. INFERENCE LOGIC


def generate_inferences(feats, osc_pred, noise_pred, dist_pred, stic_prob):
    """
    Generate 2-3 actionable inferences using both ML predictions AND feature analysis.
    """
    inferences = []

    # Extract features
    pv_std = feats.get("pv_std", 0)
    op_flat_ratio = feats.get("op_flat_ratio", 0)
    dominant_freq = feats.get("dominant_freq", 0)
    dominant_power_ratio = feats.get("dominant_power_ratio", 0)
    high_freq_ratio = feats.get("high_freq_ratio", 0)
    autocorr_peak_count = feats.get("autocorr_peak_count", 0)
    autocorr_first_peak_lag = feats.get("autocorr_first_peak_lag", 0)
    pv_op_corr = feats.get("pv_op_corr", 0)
    error_op_corr = feats.get("error_op_corr", 0)
    sp_std = feats.get("sp_std", 0)
    pv_spike_count = feats.get("pv_spike_count", 0)
    zero_crossings = feats.get("zero_crossings", 0)
    pv_reversal_count = feats.get("pv_reversal_count", 0)
    op_error_crosscorr_lag = feats.get("op_error_crosscorr_lag", 0)
    mae = feats.get("mae", 0)

    # ========================================
    # PRIORITY 1: STICTION (Most Critical)
    # ========================================
    if stic_prob > 0.6:
        # Use features to characterize stiction severity
        if op_flat_ratio > 0.2:
            inferences.append(
                f"Severe valve stiction detected (probability: {stic_prob:.0%}, flat OP segments: {op_flat_ratio:.0%}). "
                f"Valve is sticking for ~{int(op_flat_ratio*100)}% of operation time. "
                "URGENT: Inspect valve actuator - clean, lubricate, or replace."
            )
        elif pv_op_corr < 0.2:
            inferences.append(
                f"High stiction ({stic_prob:.0%}) with very low PV-OP correlation ({pv_op_corr:.2f}). "
                "Valve output not affecting process - possible mechanical failure or complete sticking. "
                "Priority action: Valve inspection required."
            )
        else:
            inferences.append(
                f"Moderate-to-high stiction risk ({stic_prob:.0%}) with some valve sticking. "
                "Recommendation: Schedule valve maintenance before performance degrades further."
            )

    elif stic_prob > 0.3:
        if osc_pred == 1 and autocorr_peak_count >= 2:
            inferences.append(
                f"Moderate stiction ({stic_prob:.0%}) combined with periodic oscillations ({autocorr_peak_count} autocorr peaks). "
                "Likely stick-slip behavior causing limit cycles. Recommend valve servicing."
            )
        elif op_flat_ratio > 0.1:
            inferences.append(
                f"Early-stage stiction detected ({stic_prob:.0%}, {int(op_flat_ratio*100)}% flat OP periods). "
                "Preventive maintenance recommended to avoid future oscillations."
            )

    # ========================================
    # PRIORITY 2: OSCILLATIONS
    # ========================================
    if osc_pred == 1:
        # Skip if already covered in stiction+oscillation case
        if not (stic_prob > 0.3 and autocorr_peak_count >= 2):

            # Characterize oscillation type using features
            if dominant_freq > 0 and dominant_freq < 0.01:
                # Low frequency = long period
                period_estimate = int(1.0 / (dominant_freq + 1e-6))
                inferences.append(
                    f"Low-frequency oscillations detected (period ≈ {period_estimate} samples). "
                    "Suggests process transport delay or slow dynamics. "
                    "Recommendation: Review process model and consider adding derivative action."
                )

            elif dominant_power_ratio > 0.25 and autocorr_peak_count >= 3:
                # Strong, clean oscillation
                inferences.append(
                    f"Strong periodic oscillations (spectral dominance: {dominant_power_ratio:.2f}, {autocorr_peak_count} harmonic peaks). "
                    "Clear limit cycle behavior. Cause: Overly aggressive controller tuning. "
                    "Action: Reduce proportional gain by 30-50% and re-tune."
                )

            elif pv_op_corr > 0.6:
                # High PV-OP correlation = controller-driven
                inferences.append(
                    f"Oscillations with strong PV-OP correlation ({pv_op_corr:.2f}) indicate controller-induced cycling. "
                    "Not caused by external factors. Recommendation: Retune PID (reduce P gain, increase D filtering)."
                )

            elif zero_crossings > 100 and pv_reversal_count > 200:
                # Many reversals = high-frequency hunting
                inferences.append(
                    f"High-frequency oscillations/hunting detected ({zero_crossings} zero crossings, {pv_reversal_count} reversals). "
                    "Controller may be responding to noise or has insufficient deadband. "
                    "Recommendation: Add measurement filtering or increase controller deadband."
                )

            else:
                # Generic oscillation
                inferences.append(
                    "Oscillatory behavior detected. Review controller tuning and check for process delays or disturbances."
                )

    # ========================================
    # PRIORITY 3: NOISE
    # ========================================
    if noise_pred == 1:
        if high_freq_ratio > 0.3:
            # Severe noise
            inferences.append(
                f"Severe measurement noise detected ({int(high_freq_ratio*100)}% high-frequency content). "
                "Sensor may be malfunctioning. "
                "URGENT: Check sensor wiring, grounding, and power supply. Add low-pass filter."
            )
        elif pv_spike_count > 10:
            # Noise with spikes
            inferences.append(
                f"Frequent measurement spikes detected ({pv_spike_count} outliers) with elevated noise. "
                "Suggests electrical interference or sensor issues. "
                "Recommendation: Inspect sensor connections and add signal conditioning."
            )
        else:
            # Moderate noise
            inferences.append(
                f"Moderate measurement noise present (high-freq ratio: {high_freq_ratio:.2f}). "
                "Consider adding exponential smoothing or low-pass filtering to PV signal."
            )

    # ========================================
    # PRIORITY 4: DISTURBANCES
    # ========================================
    if dist_pred == 1:
        if sp_std > 0.2:
            # Frequent SP changes
            inferences.append(
                f"Frequent setpoint changes detected (SP variability: {sp_std:.2f}). "
                "Loop is tracking moving target. Consider feedforward control or cascade strategy if upstream process causes changes."
            )
        elif pv_spike_count > 8:
            # Sudden disturbances
            inferences.append(
                f"Multiple sudden disturbances detected ({pv_spike_count} spikes in PV). "
                "External load changes affecting process. "
                "Recommendation: Investigate upstream sources and implement disturbance rejection tuning."
            )
        else:
            # Generic disturbance
            inferences.append(
                "External disturbances affecting loop performance. Investigate upstream process variability or load changes."
            )

    # ========================================
    # PRIORITY 5: HIGH VARIABILITY (No Clear Cause)
    # ========================================
    if pv_std > 0.5 and not osc_pred and not noise_pred and stic_prob < 0.3:
        if sp_std > 0.1:
            inferences.append(
                f"High process variability (PV std: {pv_std:.3f}) appears driven by setpoint changes (SP std: {sp_std:.2f}). "
                "Normal operation if tracking requirements. Otherwise, reduce SP variation frequency."
            )
        else:
            inferences.append(
                f"High process variability (PV std: {pv_std:.3f}) without clear oscillation, noise, or stiction. "
                "Investigate process-level issues: upstream disturbances, nonlinearity, or unmeasured load changes."
            )

    # ========================================
    # PRIORITY 6: POOR TRACKING
    # ========================================
    if mae > 0.15 and not osc_pred and stic_prob < 0.3:
        if abs(error_op_corr) < 0.2:
            inferences.append(
                f"Poor setpoint tracking (MAE: {mae:.3f}) with weak error-OP correlation. "
                "Controller not responding appropriately to errors. Check controller mode (manual vs auto) or tune integral action."
            )
        else:
            inferences.append(
                f"High steady-state tracking error (MAE: {mae:.3f}). "
                "Controller underperforming - increase proportional or integral gain."
            )

    # ========================================
    # PRIORITY 7: LAG ISSUES
    # ========================================
    if abs(op_error_crosscorr_lag) > 20:
        lag_direction = "leads" if op_error_crosscorr_lag < 0 else "lags"
        inferences.append(
            f"Significant timing mismatch: controller output {lag_direction} error by {abs(op_error_crosscorr_lag)} samples. "
            "Investigate control loop scan time, valve response delays, or measurement delays."
        )

    # ========================================
    # PRIORITY 8: GOOD PERFORMANCE
    # ========================================
    if not inferences:
        if pv_std < 0.15 and mae < 0.05:
            # Primary: Full Text (Status + Rec)
            inferences.append(
                f"Loop shows excellent performance (PV std: {pv_std:.3f}, tracking error: {mae:.3f}). "
                "Well-tuned and stable. Maintain current settings."
            )
            # Secondary: Rec only
            inferences.append("Well-tuned and stable. Maintain current settings.")
        else:
            # Primary: Full Text (Status + Rec)
            inferences.append(
                "Loop behavior is stable with no critical performance issues observed. Routine monitoring is recommended."
            )
            # Secondary: Rec only
            inferences.append("Routine monitoring is recommended.")

    # ========================================
    # LIMIT TO TOP 2 INFERENCES
    # ========================================
    return inferences[:2]


# ==========================================
# 2. FEATURE CALCULATION HELPERS
# ==========================================

def calculate_adhoc_features(df):
    """
    Computes only the features NOT available/compatible in standard features.py
    """
    # Raw Data
    pv = df['PV'].values
    sp = df['SP'].values
    op = df['OP'].values
    N = len(pv)
    error = pv - sp

    adhoc = {}

    # 1. SP Variability
    adhoc["sp_std"] = np.std(sp)

    # 2. PV Spike Count
    # Definition: Sudden changes > k * std(diff)
    d_pv = np.diff(pv)
    d_pv_std = np.std(d_pv)
    if d_pv_std > 1e-6:
        spike_thresh = 5 * d_pv_std # User suggested k, picking 5 for "Sudden"
        adhoc["pv_spike_count"] = np.sum(np.abs(d_pv) > spike_thresh)
    else:
        adhoc["pv_spike_count"] = 0

    # 3. PV Reversal Count
    # Sign change in diff
    # Use signbit to detect crossing 0
    sign_changes = np.diff(np.signbit(d_pv))
    adhoc["pv_reversal_count"] = np.count_nonzero(sign_changes)

    # 4. Zero Crossing Count (Raw)
    # features.py returns ratio. User wants count > 100.
    error_centered = error - np.mean(error)
    zc_count = len(np.where(np.diff(np.signbit(error_centered)))[0])
    adhoc["zero_crossings"] = zc_count

    # 5. High Frequency Ratio
    # FFT power in top 30% of spectrum
    f, Pxx = signal.welch(error_centered, nperseg=min(256, N))
    if len(Pxx) > 0:
        total_power = np.sum(Pxx)
        # Top 30% frequencies
        split_idx = int(0.7 * len(Pxx)) # Lower 70% vs Upper 30%
        high_freq_power = np.sum(Pxx[split_idx:])
        adhoc["high_freq_ratio"] = high_freq_power / total_power if total_power > 0 else 0
    else:
        adhoc["high_freq_ratio"] = 0

    # 6. Error-OP Correlation
    # Pearson
    if len(error) > 1 and np.std(error) > 1e-9 and np.std(op) > 1e-9:
        adhoc["error_op_corr"] = np.corrcoef(error, op)[0, 1]
    else:
        adhoc["error_op_corr"] = 0

    # 7. OP-Error Cross-Corr Lag
    if N > 10 and np.std(error) > 1e-9 and np.std(op) > 1e-9:
        # Standardize
        e_norm = (error - np.mean(error)) / np.std(error)
        o_norm = (op - np.mean(op)) / np.std(op)
        cc = signal.correlate(e_norm, o_norm, mode='full')
        lags = signal.correlation_lags(len(e_norm), len(o_norm), mode='full')
        idx_max = np.argmax(np.abs(cc))
        adhoc["op_error_crosscorr_lag"] = lags[idx_max]
    else:
        adhoc["op_error_crosscorr_lag"] = 0
        
    return adhoc


def load_feature_csv(loop_id):
    """
    Loads pre-computed feature CSV from debugs/features/
    """
    # Try finding file
    # Pattern: features_loop_{id}.csv OR features_loop_TEST_{id}.csv.
    
    debug_dir = config.DEBUG_DIR / "features"
    fname = f"features_loop_TEST_{loop_id}.csv"
    fpath = debug_dir / fname
    
    if not fpath.exists():
         # Fallback
         fpath = debug_dir / f"features_loop_{loop_id}.csv"
         
    if fpath.exists():
        try:
             df = pd.read_csv(fpath)
             if not df.empty:
                 return df.iloc[0].to_dict() # Return first row as dict
        except:
             pass
             
    print(f"Warning: Debug features matching {loop_id} not found.")
    return {}

def run_feature_generation():
    logger.info("=== Generating Advanced Inferences ===")
    
    input_csv = config.OUTPUTS_DIR / 'test_results_categorized.csv'
    
    if not input_csv.exists():
        logger.error(f"Error: {input_csv} not found.")
        return

    df_results = pd.read_csv(input_csv)
    logger.info(f"Propagating inferences for {len(df_results)} loops...")
    
    new_inferences = []
    
    for idx, row in df_results.iterrows():
        loop_id = str(row['Loop_ID'])
        
        # 1. Load Raw Data (Reuse data_loader logic or simple pandas)
        # Search in TEST_DATA_DIR
        csv_files = glob.glob(str(config.TEST_DATA_DIR / "*.csv"))
        loop_file = None
        for f in csv_files:
            if f"loop_{loop_id}_loop.csv" in f or f"{loop_id}.csv" in f:
                loop_file = f
                break
        
        if not loop_file:
            print(f"Skipping {loop_id}: Raw data not found.")
            new_inferences.append(("Data Missing", ""))
            continue
            
        try:
            # Simple Load
            raw_df = pd.read_csv(loop_file)
            # Standardize cols
            raw_df.columns = [c.strip().lower() for c in raw_df.columns]
            rename_map = {'pv_1': 'PV', 'sp_1': 'SP', 'op_1': 'OP'}
            raw_df.rename(columns=rename_map, inplace=True)
            # Ensure standard cols
            rename_std = {'pv': 'PV', 'sp': 'SP', 'op': 'OP'}
            raw_df.rename(columns=rename_std, inplace=True)
            
            raw_df = raw_df[['PV', 'SP', 'OP']].dropna()
        except:
             print(f"Skipping {loop_id}: Read Error.")
             new_inferences.append(("Read Error", ""))
             continue

        # 2. Compute Ad-Hoc Features
        adhoc_feats = calculate_adhoc_features(raw_df)
        
        # 3. Load Existing Features
        # Try to find mapped loop id (101 -> 101)
        existing_feats_raw = load_feature_csv(loop_id)
        
        # 4. Construct Inference Feature Vector
        # Map Existing -> Target Name
        infer_feats = {
            "pv_std": existing_feats_raw.get('pv_std', 0),
            "op_flat_ratio": existing_feats_raw.get('op_flatness_ratio', 0),
            "dominant_freq": existing_feats_raw.get('psd_dom_freq', 0),
            "dominant_power_ratio": existing_feats_raw.get('psd_peak_ratio', 0),
            "autocorr_peak_count": existing_feats_raw.get('acf_num_peaks', 0),
            "autocorr_first_peak_lag": existing_feats_raw.get('acf_peak_lag', 0),
            "pv_op_corr": existing_feats_raw.get('cc_max_val', 0), # Approx
            "mae": existing_feats_raw.get('error_mean_abs', 0),
            
            # Ad-Hoc
            "sp_std": adhoc_feats.get('sp_std', 0),
            "pv_spike_count": adhoc_feats.get('pv_spike_count', 0),
            "zero_crossings": adhoc_feats.get('zero_crossings', 0),
            "pv_reversal_count": adhoc_feats.get('pv_reversal_count', 0),
            "high_freq_ratio": adhoc_feats.get('high_freq_ratio', 0),
            "error_op_corr": adhoc_feats.get('error_op_corr', 0),
            "op_error_crosscorr_lag": adhoc_feats.get('op_error_crosscorr_lag', 0)
        }
        
        
        osc_flag = 1 if row['oscillations'] >= 0.3 else 0
        noise_flag = 1 if row['noise'] >= 0.3 else 0
        dist_flag = 1 if row['disturbances'] >= 0.3 else 0
        stiction_prob = row['probability_of_stiction'] # 0-1
        
        # 6. Generate Inference
        inferences = generate_inferences(
            feats=infer_feats,
            osc_pred=osc_flag,
            noise_pred=noise_flag,
            dist_pred=dist_flag,
            stic_prob=stiction_prob
        )
        
        # Split
        primary = inferences[0] if len(inferences) > 0 else ""
        secondary = inferences[1] if len(inferences) > 1 else "" # Only top 2
        
        new_inferences.append((primary, secondary))
        
    # Append to DataFrame
    df_results['detailed_primary_inference'] = [x[0] for x in new_inferences]
    df_results['detailed_secondary_inference'] = [x[1] for x in new_inferences]
    
    # ==========================================
    # 7. ADD VARIANCE LEVEL (User Logic)
    # ==========================================
    logger.info("Computing Variance Thresholds from Training Data...")
    low_thr, high_thr = get_training_variance_thresholds()
    logger.info(f"Variance Thresholds: Low<{low_thr:.4f}, High>{high_thr:.4f}")
    
    def get_var_category(val):
        if val < low_thr: return "Low"
        elif val < high_thr: return "Medium"
        return "High"
        
    df_results['Variance Level'] = df_results['variance'].apply(get_var_category)
    
    # ==========================================
    # 8. FINAL FORMATTING
    # ==========================================
    # Rename Columns
    # Map Category Columns -> User Names
    rename_map = {
        'Loop_ID': 'Loop No',
        'oscillation_category': 'Oscillatory?',
        'noise_category': 'Noisy',
        'disturbance_category': 'Disturbance',
        'variance': 'Variance',
        # 'Variance Level' is already set
        'probability_of_stiction': 'Sticky Valve probability (%)',
        'detailed_primary_inference': 'First Inference',
        'detailed_secondary_inference': 'Second Inference'
    }
    
    # Scaling Stiction to %
    df_results['probability_of_stiction'] = df_results['probability_of_stiction'] * 100.0
    
    # Rename
    df_final = df_results.rename(columns=rename_map)
    
    # Select Columns
    final_cols = [
        'Loop No',
        'Oscillatory?',
        'Noisy',
        'Disturbance',
        'Variance',
        'Variance Level',
        'Sticky Valve probability (%)',
        'First Inference',
        'Second Inference'
    ]
    
    # Ensure all exist
    df_final = df_final[final_cols]
    
    out_csv = config.OUTPUTS_DIR / 'test_results_with_inferences.csv'
    out_xlsx = config.OUTPUTS_DIR / 'test_results_with_inferences.xlsx'
    
    df_final.to_csv(out_csv, index=False)
    try:
        df_final.to_excel(out_xlsx, index=False)
    except Exception as e:
        logger.error(f"Failed to save Excel (Permission denied?): {e}")
        
    logger.info(f"Saved Final Results to {out_csv}")

def get_training_variance_thresholds():
    """
    Collects variance (PV std) from ALL training loops to define thresholds.
    """
    train_dir = config.TRAINING_LOOPS_DIR
    files = glob.glob(str(train_dir / "*.csv"))
    
    variances = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # Standardize
            cols = [c.strip().lower() for c in df.columns]
            df.columns = cols
            
            # Find PV
            pv_col = next((c for c in df.columns if 'pv' in c), None)
            if pv_col:
                pv_std = np.std(df[pv_col])
                variances.append(pv_std)
        except:
            continue
            
    if not variances:
        return 0.1, 1.0 # Default fallback
        
    var_series = pd.Series(variances)
    low_thr = var_series.quantile(0.33)
    high_thr = var_series.quantile(0.66)
    
    return low_thr, high_thr
    
    logger.info(f"\nResults with Advanced Inferences saved to:")
    logger.info(f" - CSV: {out_csv}")
    logger.info(f" - Excel: {out_xlsx}")
    
    # Preview
    logger.info("\n[Preview 101]")
    logger.info(df_results[df_results['Loop_ID'].astype(str)=='101'][['Loop_ID', 'detailed_primary_inference']].to_string(index=False))

if __name__ == "__main__":
    run_feature_generation()
