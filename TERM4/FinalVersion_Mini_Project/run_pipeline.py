import sys
import os
import time
from pathlib import Path
import numpy as np
np.random.seed(42)

# Ensure the root directory is in sys.path
# This allows running 'python run_pipeline.py' from the root
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(ROOT_DIR))

try:
    from src import config
    from src.main import run_training_pipeline
    from src.inspect_labels import run_inspection
    from src.predict_test import run_test_inference
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import project modules. {e}")
    print("Ensure you are running this script from the project root directory.")
    sys.exit(1)

def print_banner(step_name):
    print(f"\n{'='*60}")
    print(f" EXECUTION: {step_name}")
    print(f"{'='*60}\n")

def main():
    start_time = time.time()
    print(f"Starting Industrial Loop Analysis Pipeline...")
    print(f"Project Root: {config.ROOT_DIR}")
    
    # Step 0: Input Analysis
    print_banner("0. Input File Analysis")
    try:
        from src.analyze_files import analyze_files
        analyze_files()
    except Exception as e:
        print(f"FAILED: Input Analysis crashed. {e}")
        # Not critical, proceed

    # Step 1: Training Pipeline (Data -> Features -> Training)
    print_banner("1. Training & Validation Pipeline")
    if config.TRAINED_MODELS_FILE.exists():
        print(f"Model: Found existing models at {config.TRAINED_MODELS_FILE}")
        print("Skipping Training Step as requested. (Delete file to force retraining)")
    else:
        # Check requirements before training
        if not config.LOOP_DESCRIPTION_FILE.exists():
            print(f"CRITICAL: Labels file not found at {config.LOOP_DESCRIPTION_FILE}")
            print("Cannot train without labels. Aborting.")
            return

        try:
            run_training_pipeline(mode='train')
        except Exception as e:
            print(f"FAILED: Training Pipeline crashed. {e}")
            import traceback
            traceback.print_exc()
            return

    # Step 2: EDA Generation (Plots for Report)
    print_banner("2. EDA Plot Generation (from Training Data)")
    try:
        from src.generate_eda import generate_eda
        generate_eda()
    except Exception as e:
        print(f"FAILED: EDA Generation crashed. {e}")

    # Step 3: Feature Importance Visualization
    print_banner("3. Feature Importance Plots (from Models)")
    try:
        from src.visualize_importance import plot_feature_importance
        plot_feature_importance()
    except Exception as e:
        print(f"FAILED: Feature Importance Plots crashed. {e}")

    # Step 4: Label Integrity Audit
    print_banner("4. Label Integrity & Count Analysis")
    try:
        run_inspection()
    except Exception as e:
        print(f"FAILED: Label Audit crashed. {e}")
        return

    # Step 5: Test Data Inference (Held-Out Evaluation)
    print_banner("5. Held-Out Test Data Inference")
    try:
        run_test_inference()
    except Exception as e:
        print(f"FAILED: Test Inference crashed. {e}")
        return

    # Step 6: Categorization
    print_banner("6. Post-Process Categorization")
    try:
        from src.post_process_categories import run_categorization
        run_categorization()
    except Exception as e:
        print(f"FAILED: Categorization crashed. {e}")
        return

    # Step 7: Advanced Inference Generation
    print_banner("7. Advanced Inference Generation")
    try:
        from src.generate_inferences import run_feature_generation
        run_feature_generation()
    except Exception as e:
        print(f"FAILED: Inference Generation crashed. {e}")
        return

    # Step 8: Physical/Validation Checks
    print_banner("8. Physical Validation Checks (No ML)")
    try:
        from src.validate_test_physics import run_physics_validation
        run_physics_validation()
    except Exception as e:
        print(f"FAILED: Physics Validation crashed. {e}")

    # Step 9: Training Data Report
    print_banner("9. Training Data Reference Report")
    try:
        from src.generate_training_report import run_training_report
        run_training_report()
    except Exception as e:
        print(f"FAILED: Training Report crashed. {e}")
        
    # Step 10: Severity Validator
    print_banner("10. Severity vs Label Alignment")
    try:
        from src.validate_label_severity import validate_severity_alignment
        validate_severity_alignment()
    except Exception as e:
        print(f"FAILED: Severity Validator crashed. {e}")

    elapsed = time.time() - start_time
    print_banner("PIPELINE COMPLETE")
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print(f"Outputs available in: {config.OUTPUTS_DIR}")

if __name__ == "__main__":
    main()
