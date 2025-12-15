import pandas as pd
import os

def clean(x):
    if isinstance(x, str):
        return x.strip().lower()
    return ""


def map_osc(t):
    t = clean(t)
    if "highly" in t and "oscill" in t: return 2
    if "oscillating sp" in t: return 2
    if "oscill" in t: return 2
    if "intermitt" in t: return 1
    if "small oscill" in t: return 1
    if "ripple" in t: return 1
    if "variance" in t: return 1
    if t == "no" or "no oscill" in t: return 0
    if "noisy" in t: return 0
    return 0


def map_noise(t):
    t = clean(t)
    if t == "yes": return 2
    if "could" in t or "may" in t: return 1
    if "no" in t: return 0
    return 0


def map_dist(t):
    t = clean(t)
    if "disturb" in t or "yes" in t: return 2
    if "may" in t: return 1
    if "no" in t: return 0
    return 0



def process_labels(label_file="././data/loop_description.xlsx"):
    """
    Loads and cleans labels from Excel.
    Returns a DataFrame with standardized column names.
    """

    df = pd.read_excel(label_file)

    # REQUIRED columns from YOU:
    # "Loop No", "Oscillatory?", "Noisy", "Disturbance", "Sticky Valve probability"

    # Rename them to internal names
    df = df.rename(columns={
        "Loop No": "loop_id",
        "Oscillatory?": "osc",
        "Noisy": "noise",
        "Disturbance": "dist",
        "Sticky Valve probability": "stiction_prob"
    })

    # Map label text → numeric classes
    df["osc_label"]   = df["osc"].apply(map_osc)
    df["noise_label"] = df["noise"].apply(map_noise)
    df["dist_label"]  = df["dist"].apply(map_dist)

    # Stiction is already numeric
    df["stiction_prob"] = df["stiction_prob"].astype(float)

    final_df = df[["loop_id", "osc_label", "noise_label", "dist_label", "stiction_prob"]]

    return final_df


# ----------------------------------------------------------
# Enable running as a standalone script
# ----------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    labels = process_labels("data/loop_description.xlsx")

    out_path = "data/labels.csv"
    labels.to_csv(out_path, index=False)

    print(f"Labels processed successfully! Saved to: {out_path}")
