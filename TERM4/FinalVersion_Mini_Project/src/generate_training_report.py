import os
import glob
import pandas as pd
import json
import sys
import pickle

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_loop_data
from features import extract_features
from severity import calc_severity_single
from models import LoopClassifier
from utils_logger import setup_logging

def run_training_report():
    print("=== Generating Training Data Report (Excluding Test Loops) ===")
    
    BASE_DIR = os.getcwd()
    from src import config
    
    INPUTS_DIR = config.TRAINING_LOOPS_DIR
    OUTPUTS_DIR = config.OUTPUTS_DIR
    STATS_PATH = OUTPUTS_DIR / 'severity_stats.json'
    MODEL_PATH = config.TRAINED_MODELS_FILE
    
    # Exclusions
    EXCLUDE_IDS = ['12', '77', '101', '131']
    
    # Load Stats
    if not os.path.exists(STATS_PATH):
        print("Error: Severity stats not found.")
        return
    with open(STATS_PATH, 'r') as f:
        severity_stats = json.load(f)
        
    # Load Models (for Stiction Score prediction)
    if not os.path.exists(MODEL_PATH):
        print("Error: Models not found.")
        return
    clf = LoopClassifier()
    clf.load(MODEL_PATH)

    # Find Files
    files = glob.glob(os.path.join(INPUTS_DIR, "*.csv"))
    print(f"Found {len(files)} total files.")
    
    results = []
    
    for csv_path in files:
        loop_filename = os.path.basename(csv_path)
        try:
            loop_id = loop_filename.split('_')[1]
        except:
            loop_id = loop_filename.replace('.csv', '')
            
        # EXCLUSION CHECK
        if loop_id in EXCLUDE_IDS:
            print(f"Skipping Test Loop: {loop_id}")
            continue
            
        # Load
        df = load_loop_data(csv_path)
        if df is None or df.empty:
            continue
            
        # Features
        feats = extract_features(df, debug=False, loop_id=loop_id)
        
        # Severity
        sev_scores = calc_severity_single(feats, severity_stats)
        feats.update(sev_scores)
        
        # Impute missing Decay Rate (Consistency with Test Pipeline)
        if 'err_acf_decay_rate' in feats and pd.isna(feats['err_acf_decay_rate']):
            feats['err_acf_decay_rate'] = 10.0
        
        # Predict (Need Stiction Score for Probability)
        X = pd.DataFrame([feats])
        X = X.fillna(0) # Safety net for SVM
        
        pred_df, _ = clf.predict(X)
        preds_dict = pred_df.iloc[0].to_dict()
        
        # Stiction Severity Correction (Standard Logic)
        stiction_score = preds_dict.get('Stiction_Score', 0)
        if stiction_score > 1.0:
            feats['stiction_severity'] = min(stiction_score / 100.0, 1.0)
        else:
            feats['stiction_severity'] = stiction_score
            
        # Inference Logic (Strict Rank-Based)
        sev_dict = {k: v for k, v in feats.items() if k.endswith('_severity')}
        
        candidates = {
            "Stiction": feats['stiction_severity'],
            "Oscillation": sev_dict.get('oscillation_severity', 0),
            "Noise": sev_dict.get('noise_severity', 0),
            "Disturbance": sev_dict.get('disturbance_severity', 0)
        }
        
        # Primary
        valid_primary = {}
        for k, v in candidates.items():
            thresh = 0.40 if k == "Stiction" else 0.30
            if v >= thresh:
                valid_primary[k] = v
        
        if valid_primary:
            primary_key = max(valid_primary, key=valid_primary.get)
        else:
            primary_key = "None"
            
        prim_map = {
            "Stiction": "High probability of valve stiction affecting loop performance.",
            "Oscillation": "Sustained oscillatory behavior detected in the control loop.",
            "Noise": "Control signal exhibits significant random noise.",
            "Disturbance": "Loop behavior indicates presence of external disturbances.",
            "None": "No dominant abnormal loop behavior detected."
        }
        primary_inf = prim_map[primary_key]
        
        # Secondary
        secondary_list = []
        for k, v in candidates.items():
            if k == primary_key: continue
            if v >= 0.25:
                secondary_list.append((k, v))
        secondary_list.sort(key=lambda x: x[1], reverse=True)
        secondary_list = secondary_list[:2]
        
        sec_map = {
            "Oscillation": "Mild to moderate oscillatory behavior is also present.",
            "Noise": "Additional noisy behavior is observed.",
            "Disturbance": "Minor disturbance effects are visible.",
            "Stiction": "Low stiction probability observed."
        }
        second_texts = [sec_map[k] for k, v in secondary_list]
        secondary_inf = " ".join(second_texts) if second_texts else "No additional significant findings."
        
        # explicit Primary Status
        def get_status(severity):
            if severity > 0.66: return "High"
            if severity > 0.33: return "Moderate"
            return "Negligible"
        
        osc_stat = get_status(sev_dict.get('oscillation_severity', 0))
        noise_stat = get_status(sev_dict.get('noise_severity', 0))
        dist_stat = get_status(sev_dict.get('disturbance_severity', 0))
        pv_std = feats.get('pv_std', 0)
        var_stat = "High Variance" if pv_std > 1.0 else "Stable Variance"
        
        primary_explicit = f"Oscillation: {osc_stat}. Noise: {noise_stat}. Disturbance: {dist_stat}. {var_stat}."
        
        # Secondary Explicit
        stich_sev = sev_dict.get('stiction_severity', 0)
        if stich_sev < 0.33: s_band = "LOW"
        elif stich_sev < 0.66: s_band = "MEDIUM"
        else: s_band = "HIGH"
        
        ev_list = []
        if feats.get('op_flatness', 0) > 0.05: ev_list.append(f"Flatness {feats.get('op_flatness',0)*100:.1f}%")
        if feats.get('op_jump_count', 0) > 2: ev_list.append(f"Jumps {feats.get('op_jump_count',0)}")
        ev_str = ", ".join(ev_list) if ev_list else "No strong evidence"
        
        # Use Stiction Score from prediction for Probability (normalized 0-100 -> 0-1)
        prob_stiction = min(max(stiction_score / 100.0, 0.0), 1.0)
        
        secondary_explicit = f"Stiction Probability: {prob_stiction*100:.1f}% (Severity: {s_band}). Evidence: {ev_str}."

        row = {
            'Loop_ID': loop_id,
            'oscillations': sev_dict.get('oscillation_severity', 0),
            'noise': sev_dict.get('noise_severity', 0),
            'disturbances': sev_dict.get('disturbance_severity', 0),
            'variance': pv_std,
            'probability_of_stiction': prob_stiction,
            'primary_inference': primary_explicit,
            'secondary_inference': secondary_explicit
        }
        results.append(row)
        
    df_res = pd.DataFrame(results)
    
    # Save
    out_csv = os.path.join(OUTPUTS_DIR, 'training_data_summary.csv')
    df_res.to_csv(out_csv, index=False)
    
    try:
        out_xls = os.path.join(OUTPUTS_DIR, 'training_data_summary.xlsx')
        df_res.to_excel(out_xls, index=False)
        print(f"Excel report saved to {out_xls}")
    except:
        pass
        
    print(f"Reference Training Data generated for {len(df_res)} loops at {out_csv}")

if __name__ == "__main__":
    setup_logging("training_report")
    run_training_report()
