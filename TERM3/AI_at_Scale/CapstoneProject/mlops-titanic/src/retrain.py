# Simulates new data arrival, reruns preprocess/train/evaluate, 
# compares against production metrics, and promotes if better; orchestrates dvc repro or direct script calls

'''
Reads new processed Parquet (e.g., from a new batch).

Trains a new model, evaluates on reference set, and if better:

Replaces models/production with the new one.

Keeps timestamped model directories for traceability.

'''

import shutil
from datetime import datetime
from pathlib import Path

from utils.logging import get_logger
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

logger = get_logger("Retrain")

REF_PARQUET = "data/processed/train_processed.parquet"
NEW_PROC_PARQUET = "data/processed/new_train_processed.parquet"  # produced by preprocessing on new raw data
PROD_DIR = "models/production"
CANDIDATE_BASE = "models/candidates"

def _evaluate_auc(model: LogisticRegressionModel, df):
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="label",
        metricName="areaUnderROC"
    )
    return evaluator.evaluate(model.transform(df))

def retrain_if_better(ref_parquet: str = REF_PARQUET,
                      new_parquet: str = NEW_PROC_PARQUET,
                      production_dir: str = PROD_DIR,
                      candidate_base: str = CANDIDATE_BASE):
    try:
        logger.info("[LOG] Starting retraining procedure")

        spark = (SparkSession.builder
                 .appName("TitanicRetrain")
                 .getOrCreate())

        ref_df = spark.read.parquet(ref_parquet)
        new_df = spark.read.parquet(new_parquet)
        logger.info(f"[LOG] Loaded reference rows={ref_df.count()} new rows={new_df.count()}")

        # Train candidate on new data
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
        candidate = lr.fit(new_df)
        logger.info("[LOG] Candidate model trained on new data")

        # Evaluate candidate on reference for comparable metric
        cand_auc = _evaluate_auc(candidate, ref_df)
        logger.info(f"[LOG] Candidate AUC on reference: {cand_auc:.4f}")

        # Load current production (if exists) and compare
        prod_auc = -1.0
        if Path(production_dir).exists():
            prod = LogisticRegressionModel.load(production_dir)
            prod_auc = _evaluate_auc(prod, ref_df)
            logger.info(f"[LOG] Current production AUC on reference: {prod_auc:.4f}")
        else:
            logger.info("[LOG] No production model found; will promote first candidate.")

        if cand_auc > prod_auc:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            cand_dir = f"{candidate_base}/model_{ts}"
            Path(cand_dir).parent.mkdir(parents=True, exist_ok=True)
            candidate.save(cand_dir)
            logger.info(f"[LOG] Saved candidate -> {cand_dir}")

            # Promote candidate to production (replace directory atomically)
            if Path(production_dir).exists():
                shutil.rmtree(production_dir)
            shutil.copytree(cand_dir, production_dir)
            logger.info(f"[LOG] Promoted candidate to production -> {production_dir}")
        else:
            logger.info("[LOG] Candidate did not outperform production; keeping current model.")

        spark.stop()
        logger.info("[LOG] Retraining completed.")
    except Exception as e:
        logger.error(f"[ERROR] Retraining failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    retrain_if_better()

