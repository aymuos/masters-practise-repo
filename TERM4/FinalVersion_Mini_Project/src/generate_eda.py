import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from src import config

def generate_eda():
    print("Generating EDA plots...")
    
    # Ensure directory exists
    eda_dir = config.OUTPUTS_DIR / 'eda'
    eda_dir.mkdir(exist_ok=True)
    
    # Load Data
    train_file = config.OUTPUTS_DIR / 'train_dataset.csv'
    if not train_file.exists():
        print(f"Train file not found: {train_file}")
        return
        
    df = pd.read_csv(train_file)
    print(f"Loaded {len(df)} rows.")
    
    # 1. Class Balance Plots
    targets = ['Oscillation_Flag', 'Noise_Flag', 'Disturbance_Flag']
    for t in targets:
        if t not in df.columns: continue
        
        plt.figure(figsize=(6, 4))
        sns.countplot(x=t, data=df)
        plt.title(f'Class Balance: {t}')
        plt.tight_layout()
        out_path = eda_dir / f'balance_{t}.png'
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

    # 2. Correlation Matrix
    # Select feature columns (exclude IDs and Targets)
    exclude = ['Loop_ID', 'Filename', 'Oscillation_Flag', 'Noise_Flag', 'Disturbance_Flag', 'Stiction_Score']
    cols = [c for c in df.columns if c not in exclude]
    # Filter for numeric
    cols = [c for c in cols if df[c].dtype in [np.float64, np.int64]]
    
    corr = df[cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    out_path = eda_dir / 'feature_correlation.png'
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_eda()
