import joblib
import pandas as pd
from pathlib import Path

from .data_loader import load_and_clean_loop
from .feat_extraction import extract_features

def interpret(label, kind):
    maps = {
        "osc": ["No oscillation", "Maybe oscillatory", "Oscillatory"],
        "noise": ["No noise", "Maybe noisy", "Noisy"],
        "dist": ["No disturbance", "Maybe disturbed", "Disturbed"],
        "var": ["Low variance", "Medium variance", "High variance"]
    }
    return maps[kind][label]


def run_inference(loop_file):

    # Load models
    model_osc = joblib.load("models/model_osc.pkl")
    model_noise = joblib.load("models/model_noise.pkl")
    model_dist = joblib.load("models/model_dist.pkl")
    model_var = joblib.load("models/model_variance.pkl")
    model_stict = joblib.load("models/model_stiction.pkl")

    df = load_and_clean_loop(loop_file)
    loop_id = Path(loop_file).stem

    feats = extract_features(df, loop_id)
    X = pd.DataFrame([feats]).drop(columns=["loop_id"])

    # Predict
    osc = int(model_osc.predict(X)[0])
    noise = int(model_noise.predict(X)[0])
    dist = int(model_dist.predict(X)[0])
    var = int(model_var.predict(X)[0])
    stict = float(model_stict.predict(X)[0])

    print("\n===== LOOP INFERENCE REPORT =====\n")
    print(f"Loop: {loop_id}")
    print(f"Oscillation: {interpret(osc,'osc')}")
    print(f"Noise: {interpret(noise,'noise')}")
    print(f"Disturbance: {interpret(dist,'dist')}")
    print(f"Variance: {interpret(var,'var')}")
    print(f"Stiction Probability: {stict:.3f}")

    print("\n--- Summary ---")
    if osc==2: print("Strong oscillatory behavior detected.")
    if noise==2: print("High noise activity detected.")
    if dist==2: print("Strong disturbances observed.")
    if var==2: print("High variance observed.")
    if stict>0.6: print("Stiction likely.")

    return {
        "oscillation": osc,
        "noise": noise,
        "disturbance": dist,
        "variance": var,
        "stiction_prob": stict
    }
