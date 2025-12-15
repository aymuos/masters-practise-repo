import pandas as pd
from src import config
from src.utils_logger import setup_logger

logger = setup_logger("analyze_files", log_file=config.LOGS_DIR / "analyze_files.log")

def analyze_files():
    excel_path = config.LOOP_DESCRIPTION_FILE
   
    import glob
    csv_files = glob.glob(str(config.TRAINING_LOOPS_DIR / "*.csv"))
    csv_path = csv_files[0] if csv_files else None

    logger.info("--- Analysis of Loop description.xlsx ---")
    if excel_path.exists():
        try:
            df_excel = pd.read_excel(excel_path)
            logger.info(f"Columns: {df_excel.columns.tolist()}")
            logger.info("First 5 rows:")
            logger.info(f"\n{df_excel.head()}")
            logger.info(f"\nNull values:\n{df_excel.isnull().sum()}")
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
    else:
        logger.error(f"File not found: {excel_path}")

    logger.info("\n--- Analysis of Loop CSV Sample ---")
    if csv_path:
        try:
            df_csv = pd.read_csv(csv_path)
            logger.info(f"File: {csv_path}")
            logger.info(f"Columns: {df_csv.columns.tolist()}")
            logger.info("First 5 rows:")
            logger.info(f"\n{df_csv.head()}")
            logger.info("\nStatistics:")
            logger.info(f"\n{df_csv.describe()}")
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
    else:
        logger.error(f"No CSV files found in {config.TRAINING_LOOPS_DIR}")

if __name__ == "__main__":
    analyze_files()
