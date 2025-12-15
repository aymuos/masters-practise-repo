import os
from pathlib import Path

# --- Dynamic Path Config ---

SRC_DIR = Path(__file__).parent.absolute()
# Go up one level to get the Project Root
ROOT_DIR = SRC_DIR.parent

# --- Subdirectories ---
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"
PLOTS_DIR = ROOT_DIR / "plots"
TEST_DATA_DIR = ROOT_DIR / "test_data"
DEBUG_DIR = ROOT_DIR / "debugs"

# --- Specific File Paths ---
TRAINING_LOOPS_DIR = INPUTS_DIR / "Training_Loops"
LOOP_DESCRIPTION_FILE = INPUTS_DIR / "Loop description.xlsx"
TEST_LOOPS_DIR = TEST_DATA_DIR / "Test_Loops"

# --- Model Artifacts ---
MODELS_DIR = OUTPUTS_DIR / "models"
TRAINED_MODELS_FILE = MODELS_DIR / "trained_models.pkl"

# --- Outputs ---
TEST_RESULTS_CSV = OUTPUTS_DIR / "test_results.csv"
SEVERITY_ALIGNMENT_CSV = OUTPUTS_DIR / "severity_alignment.csv"

# Ensure crucial directories exist
def ensure_dirs():
    for d in [OUTPUTS_DIR, LOGS_DIR, PLOTS_DIR, MODELS_DIR, DEBUG_DIR]:
        d.mkdir(exist_ok=True, parents=True)

# Run on import
ensure_dirs()

print(f"[CONFIG] Project Root detected at: {ROOT_DIR}")
