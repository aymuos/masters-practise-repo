import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def train_all_models(features_file):

    df = pd.read_csv(features_file)

    # Create variance class (LOW/MID/HIGH)
    df["variance_class"] = pd.qcut(df["variance_level"], 3, labels=[0,1,2])

    drop_cols = ["loop_id","osc_label","noise_label","dist_label",
                 "stiction_prob","variance_class"]

    feature_cols = df.columns.difference(drop_cols)

    X = df[feature_cols]

    y_osc  = df["osc_label"]
    y_noise = df["noise_label"]
    y_dist  = df["dist_label"]
    y_var   = df["variance_class"]
    y_stict = df["stiction_prob"]

    # Train-validation split for oscillation only
    X_train, X_val, y_osc_train, y_osc_val = train_test_split(
        X, y_osc, test_size=0.2, stratify=y_osc, random_state=42
    )

    # ------------------------------
    # MODEL DEFINITIONS BASED ON TASK
    # ------------------------------

    # 1. Oscillation → XGBoost Classifier
    model_osc = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )

    # 2. Noise → Random Forest
    model_noise = RandomForestClassifier(
        n_estimators=300, 
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    # 3. Disturbance → Random Forest
    model_dist = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    # 4. Variance → Random Forest
    model_var = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    # 5. Stiction → XGBoost Regressor
    model_stict = xgb.XGBRegressor(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse"
    )

    # ------------------------------
    # 5-FOLD CROSS VALIDATION
    # ------------------------------

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nCross-validating Oscillation model...")
    cv_osc = cross_val_score(model_osc, X, y_osc, cv=skf, scoring="f1_macro")
    print("F1 Macro:", cv_osc.mean())

    print("\nCross-validating Noise model...")
    cv_noise = cross_val_score(model_noise, X, y_noise, cv=skf, scoring="f1_macro")
    print("F1 Macro:", cv_noise.mean())

    print("\nCross-validating Disturbance model...")
    cv_dist = cross_val_score(model_dist, X, y_dist, cv=skf, scoring="f1_macro")
    print("F1 Macro:", cv_dist.mean())

    print("\nCross-validating Variance model...")
    cv_var = cross_val_score(model_var, X, y_var, cv=skf, scoring="f1_macro")
    print("F1 Macro:", cv_var.mean())

    # ------------------------------
    # TRAIN FINAL MODELS ON FULL DATA
    # ------------------------------

    model_osc.fit(X, y_osc)
    model_noise.fit(X, y_noise)
    model_dist.fit(X, y_dist)
    model_var.fit(X, y_var)
    model_stict.fit(X, y_stict)

    # Save models
    joblib.dump(model_osc,  "models/model_osc.pkl")
    joblib.dump(model_noise,"models/model_noise.pkl")
    joblib.dump(model_dist, "models/model_dist.pkl")
    joblib.dump(model_var,  "models/model_variance.pkl")
    joblib.dump(model_stict,"models/model_stiction.pkl")

    print("\nAll models trained and saved successfully!")
