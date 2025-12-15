import os
import glob
import pandas as pd
import pickle
import json
import warnings
import sys

# Add src to path if needed (running from root)
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_loop_data
from features import extract_features
from models import LoopClassifier
from reporter import plot_loop, plot_severity_bars
from inference import generate_inferences
from severity import calc_severity_single
from utils_logger import setup_logging

warnings.filterwarnings('ignore')
import numpy as np
np.random.seed(42)

def run_test_inference():
    print("=== Starting Test Data Inference (No Retraining) ===")
    
    # Paths
    BASE_DIR = os.getcwd()
    TEST_DIR = os.path.join(BASE_DIR, 'test_data')
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
    
    # Artifact paths (MUST EXIST)
    from src import config
    
    # Artifact paths (Using Config)
    MODEL_PATH = config.TRAINED_MODELS_FILE
    STATS_PATH = config.OUTPUTS_DIR / 'severity_stats.json'
    
    # Output paths
    TEST_PLOTS_DIR = os.path.join(TEST_DIR, 'debug_plots')
    os.makedirs(TEST_PLOTS_DIR, exist_ok=True)
    
    # 1. Load Resources
    print(f"Loading models from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained models not found. Run analysis first.")
        return

    clf = LoopClassifier()
    clf.load(MODEL_PATH)
    
    print(f"Loading severity stats from {STATS_PATH}...")
    if not os.path.exists(STATS_PATH):
        print("Error: Severity stats not found. Run analysis first.")
        return
        
    with open(STATS_PATH, 'r') as f:
        severity_stats = json.load(f)
        
    # 2. Find Test Files
    test_files = glob.glob(os.path.join(TEST_DIR, "*.csv"))
    print(f"Found {len(test_files)} CSV files in {TEST_DIR}")
    
    results = []
    
    for csv_path in test_files:
        loop_filename = os.path.basename(csv_path)
        # Extract Loop ID from filename if possible (e.g. loop_101_loop.csv -> 101)
        # If not standard, use filename
        try:
            # Assuming format loop_{id}_loop.csv
            loop_id = loop_filename.split('_')[1] 
        except:
            loop_id = loop_filename.replace('.csv', '')
            
        print(f"Processing Loop: {loop_id} ({loop_filename})...")
        
        # A. Load Data
        df = load_loop_data(csv_path)
        if df is None or df.empty:
            print(f"  Skipping {loop_id}: Invalid data.")
            continue
            
        # B. Extract Features
        # Note: debug=True creates internal debug CSVs if configured, strictly we might not need them but user asked for "all debugs"
        feats = extract_features(df, debug=True, loop_id=f"TEST_{loop_id}")
        
        # C. Compute Severity (Using TRAINING stats)
        sev_scores = calc_severity_single(feats, severity_stats)
        feats.update(sev_scores)
        
        # Impute missing Decay Rate (Test Time)
        # Use constant 10.0 (Very High Decay = Stable) to match training logic of "max observed"
        if 'err_acf_decay_rate' in feats and pd.isna(feats['err_acf_decay_rate']):
            feats['err_acf_decay_rate'] = 10.0
            
        # Define sev_dict early for usage in Inference
        sev_dict = {k: v for k, v in feats.items() if k.endswith('_severity')}
        
        # D. Predict
        X_test = pd.DataFrame([feats])
        pred_df, prob_df = clf.predict(X_test)
        
        # Parsing predictions
        preds_dict = {}
        for col in pred_df.columns:
            preds_dict[col] = pred_df.iloc[0][col]
            
        # FIX: Stiction Severity from Prediction
        stiction_score = preds_dict.get('Stiction_Score', 0)
        # Normalize 0-100 -> 0-1
        if stiction_score > 1.0:
            feats['stiction_severity'] = min(stiction_score / 100.0, 1.0)
        else:
            feats['stiction_severity'] = stiction_score
            
        # E. Strict Automatic Inference Logic (Integrated Evidence)
        
        # 3. Define Severity Bands
        def severity_band_text(x):
            if x >= 0.6: return "High"
            elif x >= 0.3: return "Medium"
            else: return "Low"
            
        def get_evidence_text(issue, feat_dict):
            if issue == "Oscillation":
                acf = feat_dict.get('acf_peak_val', 0)
                zc = feat_dict.get('error_zero_crossings', 0)
                return f"(Evidence: ACF={acf:.2f}, ZC={zc:.2f})"
            elif issue == "Noise":
                ent = feat_dict.get('spectral_entropy', 0)
                return f"(Evidence: Entropy={ent:.2f})"
            elif issue == "Stiction":
                flat = feat_dict.get('op_flatness', 0)
                jumps = feat_dict.get('op_jump_count', 0)
                return f"(Evidence: Flatness={flat:.2f}, Jumps={jumps:.3f})"
            return ""

        # 1. Inputs & Candidates
        severities = {
            "Oscillation": sev_dict.get('oscillation_severity', 0),
            "Noise": sev_dict.get('noise_severity', 0),
            "Disturbance": sev_dict.get('disturbance_severity', 0),
            "Stiction": feats['stiction_severity']
        }
        
        # STEP A — Identify dominant behavior
        primary_issue = max(severities, key=severities.get)
        primary_severity = severities[primary_issue]
        
        # STEP B — Generate PRIMARY inference + Evidence
        band_str = severity_band_text(primary_severity)
        
        if primary_issue == "Stiction":
            base_text = f"{band_str} probability of valve stiction detected."
        elif primary_issue == "None":
            base_text = "No dominant abnormal behavior detected."
        else:
            base_text = f"{band_str} {primary_issue.lower()} behavior detected."
            
        if primary_severity >= 0.3 or primary_issue == "Stiction":
            ev = get_evidence_text(primary_issue, feats)
            primary_text = f"{base_text} {ev}"
        else:
            primary_text = base_text
            
        # STEP C — Generate SECONDARY inferences + Evidence
        # User REQ: "write about noise and disturbance in every secondary inference... if there is no noise... say so"
        # Strategy: Iterate through ALL non-dominant issues and report status.
        
        secondary_items = []
        # Fixed order for consistency: Osc, Noise, Dist, Stiction
        check_order = ["Oscillation", "Noise", "Disturbance", "Stiction"]
        
        for issue in check_order:
            if issue == primary_issue: continue
            
            sev = severities[issue]
            ev_sec = get_evidence_text(issue, feats)
            
            if sev >= 0.3:
                base_sec = f"Mild {issue.lower()} characteristics are also observed."
            elif sev >= 0.05:
                base_sec = f"Low {issue.lower()} levels detected."
            else:
                base_sec = f"Negligible {issue.lower()} detected."
                
            secondary_items.append(f"{base_sec} {ev_sec}")
                
        secondary_text = " ".join(secondary_items)
        
        # Combine (Use 2 parts only)
        inf_text = f"{primary_text} ; {secondary_text}" 
        print(f"  -> Inferences: {inf_text}")
        
        # F. Plotting
        # Main Loop Plot
        detect_str = " | ".join([f"{k}:{v}" for k,v in preds_dict.items() if (v==1 or (k=='Stiction_Score' and v>10))])
        if not detect_str: detect_str = "Normal"
        
        plot_loop(df, f"TEST_{loop_id}", TEST_PLOTS_DIR, detect_str, key_features=feats)
        
        # Capture Variance (PV Standard Deviation) for the report
        variance_val = feats.get('pv_std', 0.0)
        
        # Capture Severity Dict for results
        sev_dict = {k: v for k, v in feats.items() if k.endswith('_severity')}
        
        # Generate Severity Plot
        plot_severity_bars(sev_dict, f"TEST_{loop_id}", TEST_PLOTS_DIR)

        # G. Store Result
        row = {
            'Loop_ID': loop_id,
            'Filename': loop_filename,
            'Inference_Text': inf_text,
            'variance': variance_val
        }
        
        # Add Predictions
        row.update(preds_dict)
        # Add Severities
        row.update(sev_dict)
        
        results.append(row)
        
    # 3. Save Summary
    res_df = pd.DataFrame(results)
    

    final_rows = []
    for idx, r in res_df.iterrows():
        # Split inferences
        full_text = r.get('Inference_Text', "")
        parts = [p.strip() for p in full_text.split(';') if p.strip()]
        
        inf_1 = parts[0] if len(parts) > 0 else "No significant findings"
        inf_2 = parts[1] if len(parts) > 1 else "No additional findings"
        
        # Normalize Stiction (Score 0-100 -> Prob 0-1)
        raw_stiction = r.get('Stiction_Score', 0)
        prob_stiction = min(max(raw_stiction / 100.0, 0.0), 1.0)
        
        new_row = {
            'Loop_ID': r['Loop_ID'],
            'oscillations': r.get('oscillation_severity', 0.0), # Value 0-1
            'noise': r.get('noise_severity', 0.0),             # Value 0-1
            'disturbances': r.get('disturbance_severity', 0.0), # Value 0-1
            'variance': r.get('variance', 0.0),                # Raw PV Std value
            'probability_of_stiction': prob_stiction,          # Value 0-1
            'primary_inference': inf_1,
            'secondary_inference': inf_2
        }
        final_rows.append(new_row)

    final_df = pd.DataFrame(final_rows)
    
    # Final Columns Order
    cols = [
        'Loop_ID', 
        'oscillations', 
        'noise', 
        'disturbances', 
        'variance', 
        'probability_of_stiction', 
        'primary_inference', 
        'secondary_inference'
    ]
    final_df = final_df[cols]
    
    out_csv = os.path.join(OUTPUTS_DIR, 'test_results.csv')
    final_df.to_csv(out_csv, index=False)
    
    # Also save as Excel
    try:
        out_xls = os.path.join(OUTPUTS_DIR, 'test_results.xlsx')
        final_df.to_excel(out_xls, index=False)
        print(f"Excel summary saved to {out_xls}")
    except Exception as e:
        print(f"Could not save Excel: {e}")

    print(f"\nTest Inference Complete. Results saved to {out_csv}")
    print(f"Plots saved to {TEST_PLOTS_DIR}")

if __name__ == "__main__":
    setup_logging("test_inference")
    run_test_inference()
