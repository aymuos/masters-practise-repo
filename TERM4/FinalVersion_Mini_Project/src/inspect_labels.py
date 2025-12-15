import pandas as pd
import os
from src import config
from src.data_loader import load_loop_description
from src.utils_logger import setup_logger

logger = setup_logger("inspect_labels", log_file=config.LOGS_DIR / "inspect_labels.log")

def run_inspection():
    logger.info(f"\n{'='*40}")
    logger.info(" STEP: Label Integrity Audit")
    logger.info(f"{'='*40}")
    
    excel_path = config.LOOP_DESCRIPTION_FILE
    
    # Use the robust loader from data_loader to include Weak flags
    targets = load_loop_description(excel_path)
    if targets is None:
        logger.error("Failed to load targets for inspection.")
        return
    
    logger.info(f"Loaded Labels from: {excel_path}")
    logger.info(f"Total Loops: {len(targets)}")
    
    # Define Column Mappings (Strong vs Weak)
    audit_cols = [
        ('Oscillation', 'Oscillation_Flag', 'Oscillation_Weak_Flag'),
        ('Noise', 'Noise_Flag', 'Noise_Weak_Flag'),
        ('Disturbance', 'Disturbance_Flag', 'Disturbance_Weak_Flag')
    ]
    
    summary_data = []
    
    for category, strong_col, weak_col in audit_cols:
        if strong_col not in targets.columns:
            continue
            
        n_strong = targets[strong_col].sum()
        # If Weak Flag exists, calculate
        if weak_col in targets.columns:
            n_weak_total = targets[weak_col].sum()
            n_pure_weak = n_weak_total - n_strong
        else:
            n_weak_total = 0
            n_pure_weak = 0
            
        print(f"\nAudit: {category}")
        print(f"  - Strong Labels ('Yes'): {n_strong}")
        print(f"  - Weak Signals ('Maybe'): {n_pure_weak}")
        print(f"  - Total Potential: {n_weak_total}")
        
        summary_data.append({
            'Category': category,
            'Strong_Count': n_strong,
            'Weak_Only_Count': n_pure_weak,
            'Total_Potential': n_weak_total
        })
        
    # Save Audit to artifact
    audit_df = pd.DataFrame(summary_data)
    out_path = config.OUTPUTS_DIR / "label_audit_summary.csv"
    audit_df.to_csv(out_path, index=False)
    print(f"\n[Audit] Summary saved to {out_path}")

if __name__ == "__main__":
    run_inspection()
