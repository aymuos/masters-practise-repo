import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_loop(df, loop_id, output_dir, detected_info="", key_features=None):
    """
    Generates and saves a plot for the loop data.
    
    Args:
        df (pd.DataFrame): Loop data with PV, SP, OP.
        loop_id (str): Identifier.
        output_dir (str): Directory to save plots.
        detected_info (str): String containing detected behaviors to add to title.
        key_features (dict): Optional dict of important feature values to display.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Construct feature string for title
    feat_str = ""
    if key_features:
        # Select key features to show
        keys_to_show = {
            'ZeroCross': key_features.get('error_zero_crossings'),
            'ACF': key_features.get('acf_peak_val'),
            'FlatOP': key_features.get('op_flatness_ratio'),
            'Entropy': key_features.get('spectral_entropy')
        }
        feat_items = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in keys_to_show.items() if v is not None]
        feat_str = " | ".join(feat_items)
        
    title_text = f"Loop {loop_id} - PV vs SP\n{detected_info}"
    if feat_str:
        title_text += f"\n[{feat_str}]"
    
    # Plot PV and SP
    if 'SP' in df.columns:
        ax1.plot(df.index, df['SP'], label='Set Point (SP)', color='green', linestyle='--')
    if 'PV' in df.columns:
        ax1.plot(df.index, df['PV'], label='Process Variable (PV)', color='blue')
        
    ax1.set_title(title_text)
    ax1.set_ylabel("Value")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot OP
    if 'OP' in df.columns:
        ax2.plot(df.index, df['OP'], label='Controller Output (OP)', color='red')
        
    ax2.set_title("Controller Output (OP)")
    ax2.set_ylabel("Output %")
    ax2.set_xlabel("Sample")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{loop_id}.png")
    plt.savefig(save_path)
    plt.close(fig)

def save_summary(summary_list, output_path):
    """
    Saves the analysis summary to a CSV file.
    
    Args:
        summary_list (list of dict): List of result dictionaries.
        output_path (str): Path to save CSV.
    """
    if not summary_list:
        print("No summary data to save.")
        return
        
    df = pd.DataFrame(summary_list)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")

def plot_class_balance(y_dict, output_dir):
    """
    Plots the class imbalance for each categorical target.
    """
    os.makedirs(output_dir, exist_ok=True)
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    for target_name, y in y_dict.items():
        if 'Score' in target_name: continue # Skip regression
        
        plt.figure(figsize=(6, 4))
        # Use a dictionary for palette to ensure consistent coloring (0=Blue/Cool, 1=Red/Warm)
        # Assuming 0 is negative/normal and 1 is positive/anomaly
        # Handle both int (0,1) and str ('0','1') just in case
        palette = {0: 'skyblue', 1: 'salmon', '0': 'skyblue', '1': 'salmon'}
        sns.countplot(x=y, palette=palette)
        plt.title(f'Class Balance: {target_name}')
        plt.xlabel('Class (0=Normal, 1=Event)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'balance_{target_name}.png'))
        plt.close()

def plot_correlation_matrix(df, output_dir):
    """
    Plots the correlation matrix of features.
    """
    os.makedirs(output_dir, exist_ok=True)
    import seaborn as sns
    import numpy as np
    
    # Drop IDs and non-numeric
    numeric_df = df.select_dtypes(include=[np.number])
    if 'Loop_ID' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Loop_ID'])
        
    plt.figure(figsize=(10, 8))
    # Calculate corr
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
    plt.close()

def plot_severity_bars(severity_dict, loop_id, output_dir):
    """
    Plots a horizontal bar chart of severity scores.
    """
    os.makedirs(output_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    
    labels = []
    values = []

    for k, v in severity_dict.items():
        if v is not None:
            # Format: 'oscillation_severity' -> 'Oscillation'
            label = k.replace('_severity', '').capitalize()
            labels.append(label)
            values.append(v)
    
    if not labels:
        return # Nothing to plot

    plt.figure(figsize=(6, 3))
    plt.barh(labels, values, color='teal')
    plt.xlim(0, 1.1) # Band is 0-1
    plt.xlabel("Severity Score")
    plt.title(f"Loop {loop_id} - Severity Overview")
    
    # Add Band zones
    # Low (0-0.33), Med (0.33-0.66), High (0.66-1.0)
    plt.axvline(x=0.33, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.66, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"severity_loop_{loop_id}.png"))
    plt.close()
