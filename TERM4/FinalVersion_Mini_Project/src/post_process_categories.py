import pandas as pd
import os
from src import config
from src.utils_logger import setup_logger

logger = setup_logger("post_process", log_file=config.LOGS_DIR / "post_process.log")

def map_oscillation_category(score):
    if score < 0.30:
        return "No oscillation"
    elif score < 0.60:
        return "Mild oscillation"
    else:
        return "Highly oscillatory"

def map_noise_category(score):
    if score < 0.30:
        return "No noise"
    elif score < 0.60:
        return "May be Noisy"
    else:
        return "Noisy"

def map_disturbance_category(score):
    if score < 0.30:
        return "No disturbance"
    elif score < 0.60:
        return "Possible disturbance"
    else:
        return "Disturbance detected"

def run_categorization():
    logger.info("=== Post-Processing: Applying Logical Categories ===")
    
    input_csv = config.OUTPUTS_DIR / 'test_results.csv'
    
    if not input_csv.exists():
        logger.error(f"Error: {input_csv} not found. Run pipeline first.")
        return

    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} rows from {input_csv}")
    
    # Apply Mappings
    logger.info("Mapping categories...")
    # Ensure columns exist (using lowercase names as per test_results.csv)
    # Col names: Loop_ID,oscillations,noise,disturbances,...
    
    df['oscillation_category'] = df['oscillations'].apply(map_oscillation_category)
    df['noise_category'] = df['noise'].apply(map_noise_category)
    df['disturbance_category'] = df['disturbances'].apply(map_disturbance_category)
    
    # Reorder Columns
    cols_order = [
        'Loop_ID',
        'oscillations',
        'oscillation_category',
        'noise',
        'noise_category',
        'disturbances',
        'disturbance_category',
        'variance',
        'probability_of_stiction',
        'primary_inference',
        'secondary_inference'
    ]
    
    # Filter for existing columns only to be safe, but we expect all
    final_cols = [c for c in cols_order if c in df.columns]
    df_final = df[final_cols]
    
    # Save Outputs
    out_csv = config.OUTPUTS_DIR / 'test_results_categorized.csv'
    out_xlsx = config.OUTPUTS_DIR / 'test_results_categorized.xlsx'
    
    df_final.to_csv(out_csv, index=False)
    df_final.to_excel(out_xlsx, index=False)
    
    logger.info(f"\nCategorized results saved to:")
    logger.info(f" - CSV: {out_csv}")
    logger.info(f" - Excel: {out_xlsx}")
    
    # Preview
    logger.info("\n[Preview]")
    logger.info(df_final.head().to_string(index=False))

if __name__ == "__main__":
    run_categorization()
