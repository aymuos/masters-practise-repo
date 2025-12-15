import pandas as pd
import numpy as np
import os
import glob
from src import config

def load_data(loops_dir=None, description_file=None):
    """
    Loads ALL loop data and ground truth labels.
    Uses paths from config.py unless overridden.
    
    Returns:
        df_dict (dict): {loop_id: pd.DataFrame}
        targets_df (pd.DataFrame): Ground truth labels
    """
    if loops_dir is None:
        loops_dir = config.TRAINING_LOOPS_DIR
    if description_file is None:
        description_file = config.LOOP_DESCRIPTION_FILE
        
    print(f"[Data Loader] Loading training loops from: {loops_dir}")
    print(f"[Data Loader] Loading labels from: {description_file}")
    
    # 1. Load Labels
    targets_df = load_loop_description(description_file)
    if targets_df is None:
        print("CRITICAL: Leading labels failed.")
        return {}, None
        
    # 2. Load Loop Files
    df_dict = {}

    # Matches 'loop_*.csv' or just '*.csv'
    csv_pattern = os.path.join(loops_dir, "*.csv")
    files = glob.glob(csv_pattern)
    
    if not files:
        print(f"Warning: No CSV files found in {loops_dir}")
        return {}, targets_df
        
    print(f"Found {len(files)} loop files. Processing...")
    
    for f in files:
        # Extract ID from filename (e.g. 'loop_102_loop.csv' -> 102)
        basename = os.path.basename(f)
        try:

            parts = basename.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                loop_id = int(parts[1])
            else:
                name_no_ext = os.path.splitext(basename)[0]
                # If name is '102', use it
                if name_no_ext.isdigit():
                    loop_id = int(name_no_ext)
                else:
                    # Skip if can't determine ID
                    continue
                    
            # Load Data
            df = load_loop_data(f)
            if df is not None:
                df_dict[loop_id] = df
                
        except Exception as e:
            print(f"Skipping file {basename}: {e}")
            
    print(f"[Results] Loaded {len(df_dict)} valid loops.")
    return df_dict, targets_df

def load_loop_data(csv_path):
    """
    Loads loop data from a CSV file.
    Expects columns: 'pv_1', 'sp_1', 'op_1' -> Mapped to PV, SP, OP.
    """
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        # Standardize column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Map known variations
        col_map = {
            'pv_1': 'PV', 'pv': 'PV',
            'sp_1': 'SP', 'sp': 'SP',
            'op_1': 'OP', 'op': 'OP'
        }
        
        df = df.rename(columns=col_map)
        
        # Verify
        if not all(c in df.columns for c in ['PV', 'SP', 'OP']):
            return None
            
        # Clean
        df = df[['PV', 'SP', 'OP']].dropna()
        if df.empty: return None
        
        return df
        
    except Exception:
        return None

def load_loop_description(excel_path):
    """
    Loads ground truth labels. Handles Strong vs Weak flags.
    """
    if not os.path.exists(excel_path):
        return None
        
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        # Clean columns (handle trailing spaces like 'Sticky Valve probability ')
        df.columns = df.columns.str.strip()
        
        targets = pd.DataFrame()
        targets['Loop_ID'] = df['Loop No']
        
        # Helper: Strong Labels (Training Targets)
        def to_strong_flag(val):
            if pd.isna(val): return 0
            s = str(val).lower().strip()
            return 1 if s in ['yes', 'year', '1', 'true', 'y', 'yes at pv'] else 0
            
        # Helper: Weak Labels (Validation/Severity Context)
        def to_weak_flag(val):
            if pd.isna(val): return 0
            s = str(val).lower().strip()
            return 1 if to_strong_flag(val) or any(k in s for k in ['may', 'could', 'possible', 'weak', 'variance', 'intermittent']) else 0
            
        if 'Oscillatory?' in df.columns:
            targets['Oscillation_Flag'] = df['Oscillatory?'].apply(to_strong_flag)
            targets['Oscillation_Weak_Flag'] = df['Oscillatory?'].apply(to_weak_flag)
            
        if 'Noisy' in df.columns:
            targets['Noise_Flag'] = df['Noisy'].apply(to_strong_flag)
            targets['Noise_Weak_Flag'] = df['Noisy'].apply(to_weak_flag)
            
        if 'Disturbance' in df.columns:
            targets['Disturbance_Flag'] = df['Disturbance'].apply(to_strong_flag)
            targets['Disturbance_Weak_Flag'] = df['Disturbance'].apply(to_weak_flag)
            
        if 'Sticky Valve probability' in df.columns:
            targets['Stiction_Score'] = pd.to_numeric(df['Sticky Valve probability'], errors='coerce').fillna(0)
            
        return targets
        
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return None
