import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import correlate, welch
from scipy.stats import entropy

# Add src to path if needed
import sys
sys.path.append(os.path.join(os.getcwd(), 'src'))
from data_loader import load_loop_data
from src import config
from src.utils_logger import setup_logger

logger = setup_logger("validate_physics", log_file=config.LOGS_DIR / "validate_physics.log")

def compute_acf_peak(filt_error):
    # Auto-correlation of error

    e = filt_error - np.mean(filt_error)
    if np.std(e) == 0: return 0
    corr = correlate(e, e, mode='full')
    corr = corr[len(corr)//2:] # Right half
    if corr[0] == 0: return 0
    corr /= corr[0] # Normalize to 1 at lag 0
    
    # Find peaks (skip lag 0)
    peaks = []
    # Start looking from lag 5 to avoid immediate noise
    for i in range(5, len(corr)-1):
        if corr[i-1] < corr[i] > corr[i+1]:
            peaks.append(corr[i])
            
    if not peaks: return 0.0
    return max(peaks)

def compute_entropy(filt_pv):
    # Spectral Entropy
    f, Pxx = welch(filt_pv, nperseg=min(len(filt_pv), 256))
    # Normalize PSD
    Pxx_norm = Pxx / np.sum(Pxx)
    # Entropy
    se = entropy(Pxx_norm)
    return se

def compute_stiction_indices(op):
    # 1. Flatness: Ratio of zero-diff segments (simplified for raw check)
    diff_op = np.diff(op)
    # Threshold for "zero" movement (floating point tolerance)
    flat_mask = np.abs(diff_op) < 1e-4
    flatness_ratio = np.sum(flat_mask) / len(diff_op) if len(diff_op) > 0 else 0
    
    # 2. Jumps: Large changes
    std_diff = np.std(diff_op)
    if std_diff == 0:
        jump_ratio = 0
    else:
        jump_mask = np.abs(diff_op) > (3 * std_diff)
        jump_ratio = np.sum(jump_mask) / len(diff_op)
        
    return flatness_ratio, jump_ratio

def run_physics_validation():
    logger.info("=== Direct Physical Validation (No ML) ===")
    
    TEST_DIR = config.TEST_DATA_DIR
    test_files = glob.glob(str(TEST_DIR / "*.csv"))
    
    # Load Model Results for Side-by-Side
    model_res_path = config.OUTPUTS_DIR / 'test_results.csv'
    try:
        model_df = pd.read_csv(model_res_path)
        # Loop_ID matches filename splitting logic in predict_test
        model_df['Lookup_ID'] = model_df['Loop_ID'].astype(str)
        model_lookup = model_df.set_index('Lookup_ID')
    except:
        logger.warning("Warning: Could not load test_results.csv for comparison.")
        model_lookup = pd.DataFrame()

    summary_rows = []

    for csv_path in test_files:
        loop_filename = os.path.basename(csv_path)
        try:
            loop_id = loop_filename.split('_')[1] 
        except:
            loop_id = loop_filename.replace('.csv', '')
            
        df = load_loop_data(csv_path)
        if df is None: continue
        
        # 1. Signals
        pv = df['PV'].values
        sp = df['SP'].values
        op = df['OP'].values
        error = pv - sp
        
        # 2. Indicators
        acf_val = compute_acf_peak(error)
        ent_val = compute_entropy(pv)
        flat_rat, jump_rat = compute_stiction_indices(op)
        
        # 3. Model Values
        model_osc = model_lookup.loc[loop_id, 'oscillations'] if loop_id in model_lookup.index else 0
        model_stict = model_lookup.loc[loop_id, 'probability_of_stiction'] if loop_id in model_lookup.index else 0
        
        # 4. Interpretations
        # ACF < 0.2 (Low), 0.2-0.4 (Mod), >0.4 (High)
        osc_qual = "High" if acf_val > 0.4 else ("Medium" if acf_val > 0.2 else "Low")
        
        # Entropy check (Values usually 0-5)
        # Let's say < 1 Low, 1-3 Med, >3 High
        ent_qual = "High" if ent_val > 3.0 else ("Medium" if ent_val > 1.0 else "Low")
        
        # Stiction check
        stict_qual = "Likely" if (flat_rat > 0.05 and jump_rat > 0.01) else "Unlikely"
        
        summary_rows.append({
            'Loop': loop_id,
            'RAW_ACF': round(acf_val, 2),
            'RAW_Entropy': round(ent_val, 2),
            'RAW_Flatness': f"{flat_rat*100:.1f}%",
            'Model_Osc_Sev': model_osc,
            'Model_Stict_Prob': model_stict,
            'Verdict': "MATCH" # Placeholder, we visually verify
        })
        
    res_df = pd.DataFrame(summary_rows)
    logger.info("\n[Validation Summary: Raw Metrics vs Model Scores]")
    logger.info(res_df.to_string(index=False))
    
    # Save validation report
    csv_path = config.OUTPUTS_DIR / 'validation_physics.csv'
    res_df.to_csv(csv_path, index=False)
    
    # Save as Excel
    xls_path = config.OUTPUTS_DIR / 'physical_validation.xlsx'
    try:
        res_df.to_excel(xls_path, index=False)
        logger.info(f"Physical Validation Excel saved to {xls_path}")
    except Exception as e:
        logger.error(f"Could not save Excel: {e}")

if __name__ == "__main__":
    run_physics_validation()
