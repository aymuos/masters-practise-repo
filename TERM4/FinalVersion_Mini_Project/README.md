# Industrial Loop Data Analysis System

This project implements an automated Machine Learning pipeline for analyzing industrial control loop data. It detects anomalies such as **Oscillations**, **Noise**, **Disturbances**, and **Valve Stiction** using a combination of Signal Processing features and ML Classifiers (Random Forest, XGBoost, SVM).

## Quick Start
 To run the **entire pipeline** (Training -> Validation -> Test Inference) in one go:
```bash
python run_pipeline.py
```
This script acts as the master orchestrator, handling dependency checks, directory setup, and sequential execution.

---

## Execution Modes
You can also run specific components of the system individually.

### 1. Training & Validation
To run only the training pipeline (loads data, extracts features, trains models, generates validation report):
```bash
python src/main.py
```
*   **Output**: `outputs/analysis_summary.csv`, `outputs/models/trained_models.pkl`

### 2. Single Loop Debugging
To deep-dive into a specific loop (prints feature values, fired rules, and generates a dedicated plot):
```bash
python src/main.py --mode debug --loop_id 102
```
*   **Output**: Console trace, `logs/main.log`, and `test_data/debug_plots/TEST_102.png` (or similar).

### 3. Label Integrity Audit
To analyze the "Strong" vs "Weak" labels in the ground truth data:
```bash
python src/inspect_labels.py
```
*   **Output**: `outputs/label_audit_summary.csv`

### 4. Test Data Inference
To run the trained models on the held-out test set (`test_data/`) without retraining:
```bash
python src/predict_test.py
```
*   **Output**: `outputs/test_results.csv`, `outputs/test_results.xlsx`

---

## Project Structure
*   **`run_pipeline.py`**: Master entry point.
*   **`src/`**: Source code.
    *   `config.py`: Centralized path configuration.
    *   `features.py`: Signal processing & feature extraction logic (ACF, PSD, Entropy, etc.).
    *   `models.py`: Machine Learning wrappers (RandomForest, XGBoost, etc.).
    *   `main.py`: Training pipeline orchestration.
    *   `predict_test.py`: Inference engine for new data.
*   **`inputs/`**: Raw CSV data and `Loop description.xlsx`.
*   **`outputs/`**: Generated reports, CSVs, and trained models.
*   **`logs/`**: Execution logs (e.g., `main.log`, `pipeline.log`).

## Requirements
*   Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn, XGBoost, OpenPyXL.
*   Install via: `pip install -r requirements.txt`
