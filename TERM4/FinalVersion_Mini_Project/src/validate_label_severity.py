import os
import pandas as pd
import json
import numpy as np
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))
from data_loader import load_loop_data, load_loop_description
from features import extract_features
from severity import calc_severity_single
from src import config
from src.utils_logger import setup_logger

logger = setup_logger("validate_severity", log_file=config.LOGS_DIR / "validate_severity.log")

def validate_severity_alignment():
    logger.info("=== Validating Severity vs Label Strength ===")
    
    INPUTS_DIR = config.TRAINING_LOOPS_DIR
    EXCEL_PATH = config.LOOP_DESCRIPTION_FILE
    STATS_PATH = config.OUTPUTS_DIR / 'severity_stats.json'
    
    # Load Stats
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
        
    # Load Labels (with Strong/Weak flags)
    labels_df = load_loop_description(EXCEL_PATH)
    if labels_df is None:
        logger.error("Failed to load labels.")
        return
        
    # Merge Features & Labels
    data = []
    
    # Iterate all CSVs to get Calc Severity
    # Use config paths
    import glob
    files = glob.glob(str(INPUTS_DIR / "*.csv"))
    logger.info(f"Processing {len(files)} loops...")
    
    for csv_path in files:
        fname = os.path.basename(csv_path)
        try:
            lid = fname.split('_')[1]
        except:
            lid = fname.replace('.csv', '')
            
        # Get Label Row
        try:
            # Loop No can be int or str
            label_row = labels_df[labels_df['Loop_ID'].astype(str) == str(lid)].iloc[0]
        except:
            # logger.debug(f"Label not found for {lid}")
            continue
            
        # Load Data
        df = load_loop_data(csv_path)
        if df is None or df.empty: continue
        
        # Calc Severity
        feats = extract_features(df, debug=False, loop_id=lid)
        sev = calc_severity_single(feats, stats)
        
        # Store info
        row = {
            'Loop_ID': lid,
            # Labels
            'Osc_Strong': label_row.get('Oscillation_Flag', 0),
            'Osc_Weak': label_row.get('Oscillation_Weak_Flag', 0),
            'Noise_Strong': label_row.get('Noise_Flag', 0),
            'Noise_Weak': label_row.get('Noise_Weak_Flag', 0),
            'Dist_Strong': label_row.get('Disturbance_Flag', 0),
            'Dist_Weak': label_row.get('Disturbance_Weak_Flag', 0),
            # Computed Severity
            'Osc_Sev': sev['oscillation_severity'],
            'Noise_Sev': sev['noise_severity'],
            'Dist_Sev': sev['disturbance_severity']
        }
        data.append(row)
        
    df_res = pd.DataFrame(data)
    
    # --- Aggregation Helper ---
    def get_category(strong, weak):
        if strong == 1: return "Strong"
        if weak == 1: return "Weak"
        return "None"
    
    summary_rows = []
    
    for metric, label_prefix in [('Oscillation', 'Osc'), ('Noise', 'Noise'), ('Disturbance', 'Dist')]:
        # Categorize
        cats = df_res.apply(lambda r: get_category(r[f'{label_prefix}_Strong'], r[f'{label_prefix}_Weak']), axis=1)
        
        # Groupby
        grouped = df_res.groupby(cats)[f'{label_prefix}_Sev'].mean()
        
        summary_rows.append({
            'Category': metric,
            'Avg Severity (Strong=1)': grouped.get('Strong', 0.0),
            'Avg Severity (Weak=1)': grouped.get('Weak', 0.0),
            'Avg Severity (None)': grouped.get('None', 0.0)
        })
        
    summary_df = pd.DataFrame(summary_rows)
    logger.info("\n=== Validation Table: Severity Alignment ===")
    logger.info(summary_df.to_string(index=False))
    
    out_path = config.OUTPUTS_DIR / 'severity_alignment.csv'
    summary_df.to_csv(out_path, index=False)
    logger.info(f"\nSaved to {out_path}")

if __name__ == "__main__":
    validate_severity_alignment()
