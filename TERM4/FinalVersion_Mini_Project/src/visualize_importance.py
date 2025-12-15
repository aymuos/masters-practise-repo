import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src import config

def plot_feature_importance(target=None):
    """
    Plots feature importance for specified target (or all if None).
    """
    targets = [target] if target else ['Oscillation_Flag', 'Noise_Flag', 'Stiction_Score']
    
    eda_dir = config.PLOTS_DIR / 'eda'
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    for t in targets:
        csv_path = config.OUTPUTS_DIR / f'feature_importance_{t}.csv'
        if not csv_path.exists():
            # print(f"Skipping {t}: Feature importance file not found.")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            # Sort just in case
            df = df.sort_values(by='importance', ascending=False).head(10)
            
            # Use logger if available, otherwise print
            print(f"\n[Feature Importance] Top 5 Features for {t}")
            print(df.head(5).to_string(index=False))
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=df, palette='viridis')
            plt.title(f'Feature Importance: {t}')
            plt.tight_layout()
            save_path = eda_dir / f'importance_{t}.png'
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.close()
        except Exception as e:
            print(f"Error plotting importance for {t}: {e}")

if __name__ == "__main__":
    plot_feature_importance()
