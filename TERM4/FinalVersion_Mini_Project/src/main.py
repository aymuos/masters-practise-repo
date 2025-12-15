import os
import sys
import json
import pandas as pd
import numpy as np
import warnings
import joblib
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

from src import config
from src.utils_logger import setup_logger

np.random.seed(42) # Ensure global reproducibility
from src.data_loader import load_data, load_loop_data
from src.features import extract_features as extract_features_single
from src.models import LoopClassifier
from src.severity import calc_severity_batch, calc_severity_single
from src.inference import generate_inferences
from src.visualize_importance import plot_feature_importance
from src.reporter import  plot_loop, save_summary, plot_class_balance, plot_correlation_matrix, plot_severity_bars

# Setup Logger
logger = setup_logger("main_pipeline", log_file=config.LOGS_DIR / "main.log")

def run_training_pipeline(mode='train', loop_id=None):
    """
    Main entry point for the Training/Validation pipeline.
    """
    logger.info("Starting Pipeline...")
    
    # --- 1. Load Data ---
    # Unified loader from data_loader.py which uses config defaults
    df_dict, targets_df = load_data()
    
    if not df_dict:
        logger.error("No loop data found. pipeline aborting.")
        return
        
    if targets_df is None or targets_df.empty:
        logger.error("No ground truth labels found.")
        return

    # --- 2. Feature Extraction ---
    logger.info("Extracting features from loop data...")
    feature_list = []
    
    loop_ids = list(df_dict.keys())
    # Debug: process limited list or all
    
    # We use the existing batch extractor if available or manual loop
    # Let's manual loop to ensure robustness and logging
    for idx, (lid, df) in enumerate(df_dict.items()):
        feats = extract_features_single(df, debug=True, loop_id=str(lid))
        if feats:
            feats['Loop_ID'] = str(lid)
            feature_list.append(feats)
            
        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_dict)} loops...")
            
    if not feature_list:
        logger.error("No features extracted.")
        return
        
    X_full = pd.DataFrame(feature_list)
    logger.info(f"Extracted features for {len(X_full)} loops.")
    
    # Save Feature Snapshot
    X_full.to_csv(config.OUTPUTS_DIR / 'features_raw.csv', index=False)
    
    # --- 3. Merge with Targets ---
    targets_df['Loop_ID'] = targets_df['Loop_ID'].astype(str)
    
    # Merge
    # Note: targets_df has "Oscillation_Flag", etc.
    data_merged = pd.merge(X_full, targets_df, on='Loop_ID', how='inner')
    logger.info(f"Merged Data Shape: {data_merged.shape}")
    
    if data_merged.empty:
        logger.error("Merged data is empty. Check Loop_ID matching.")
        return
        
    # Save Dataset
    data_merged.to_csv(config.OUTPUTS_DIR / 'train_dataset.csv', index=False)
    
    # Impute Missing Values (specifically for Decay Rate)
    if 'err_acf_decay_rate' in data_merged.columns:
        # Fill with max observed decay (safest assumption: if no peaks, it decayed instantly)
        max_decay = data_merged['err_acf_decay_rate'].max()
        if pd.isna(max_decay): max_decay = 1.0 # Fallback if all are NaN
        data_merged['err_acf_decay_rate'].fillna(max_decay, inplace=True)
        logger.info(f"Imputed 'err_acf_decay_rate' NaNs with {max_decay:.4f}")

    # --- 4. Severity Calculation ---
    logger.info("Computing Robust Severity Scores...")
    data_merged, severity_stats = calc_severity_batch(data_merged)
    
    # Save stats
    with open(config.OUTPUTS_DIR / 'severity_stats.json', 'w') as f:
        json.dump(severity_stats, f, indent=4)
        
    # Save debug features
    data_merged.to_csv(config.DEBUG_DIR / 'complete_training_features.csv', index=False)
    
    # --- 5. Training ---
    # Prepare Data
    target_cols = ['Oscillation_Flag', 'Noise_Flag', 'Disturbance_Flag', 'Stiction_Score']
    available_targets = [c for c in target_cols if c in data_merged.columns]
    
    # Exclude non-feature cols
    exclude_cols = ['Loop_ID'] + [c for c in data_merged.columns if c.endswith('_severity')] + available_targets
    # Also exclude Weak flags if present
    exclude_cols += [c for c in data_merged.columns if '_Weak_' in c]
    
    feature_cols = [c for c in X_full.columns if c not in exclude_cols]
    
    X = data_merged[feature_cols]
    y_dict = {t: data_merged[t] for t in available_targets}
    
    # EDA
    plot_class_balance(y_dict, config.PLOTS_DIR / "eda")
    plot_correlation_matrix(X, config.PLOTS_DIR / "eda")
    
    # Train
    logger.info("Training Models (5-Fold CV)...")
    clf = LoopClassifier()
    best_results, cv_predictions, full_log = clf.train_and_evaluate_cv(X, y_dict, k=5)
    
    # Save Artifacts
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(config.OUTPUTS_DIR / 'model_comparison.json', 'w') as f:
        json.dump(full_log, f, indent=4, cls=NumpyEncoder)
        
    clf.save(config.TRAINED_MODELS_FILE)
    logger.info(f"Models saved to {config.TRAINED_MODELS_FILE}")
    
    # --- 6. Generate Validation Report ---
    # (Matches original run_analysis logic)
    print("Generating Validation Inferences...")
    
    # Create map for easy plotting
    loop_data_map = df_dict
    
    summary_results = []
    preds_df = cv_predictions
    
    # Index alignment is tricky if X was subset. 
    # But here X is derived from data_merged and rows are aligned.
    # Safe to iterate data_merged and use index
    
    for idx, row in data_merged.iterrows():
        loop_id = row['Loop_ID'] # String
        int_id = int(loop_id) if loop_id.isdigit() else loop_id
        
        # Get preds
        loop_preds = {t: preds_df.loc[idx, t] for t in preds_df.columns}
        
        # Features map
        row_features = {c: row[c] for c in feature_cols if c in row}
        
        # Inferences
        inferences, _ = generate_inferences(loop_preds, row_features, loop_id)
        
        # Plot Severity
        sev_dict = {k: row[k] for k in row.index if str(k).endswith('_severity')}
        plot_severity_bars(sev_dict, str(loop_id), config.OUTPUTS_DIR / "severity_plots")
        
        # Store
        res = {'Loop_ID': loop_id}
        res.update(loop_preds)
        res['Inference'] = "; ".join(inferences.values())
        
        # Add Actuals
        for k in available_targets:
            res[f'Actual_{k}'] = row[k]
            
        summary_results.append(res)
        
    # Save Summary
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(config.OUTPUTS_DIR / 'analysis_summary.csv', index=False)
    
    logger.info("Pipeline Complete.")
    return clf

def run_debug(loop_id):
    """
    Single Loop Debug Mode
    """
    logger.info(f"Starting Debug for Loop {loop_id}")
    
    # 1. Load File
    # Try finding file
    # We need to construct path. Using standard convention
    # But data_loader.load_data finds it dynamically.
    # Let's try to match it manually or verify
    fname = f"loop_{loop_id}_loop.csv"
    fpath = config.TRAINING_LOOPS_DIR / fname
    if not fpath.exists():
         # Try just ID
         fpath = config.TRAINING_LOOPS_DIR / f"{loop_id}.csv"
         
    if not fpath.exists():
        logger.error(f"File for loop {loop_id} not found.")
        return
        
    df = load_loop_data(fpath)
    if df is None:
        logger.error(f"Failed to load data for {loop_id}")
        return
        
    # 2. Features
    feats = extract_features_single(df, debug=True, loop_id=str(loop_id))
    
    # 3. Model Predict
    if config.TRAINED_MODELS_FILE.exists():
        clf = LoopClassifier()
        clf.load(config.TRAINED_MODELS_FILE)
        X_df = pd.DataFrame([feats])
        pred_df, prob_df = clf.predict(X_df)
        
        print("\n[Predictions]")
        print(pred_df)
        print("\n[Probabilities]")
        print(prob_df)
    else:
        logger.warning("No models found. Skipping prediction.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'debug'], default='train')
    parser.add_argument('--loop_id', type=str, help="Loop ID for debug")
    args = parser.parse_args()
    
    if args.mode == 'debug':
        run_debug(args.loop_id)
    else:
        run_training_pipeline()
