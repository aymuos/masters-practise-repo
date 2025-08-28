# Trains classifier(s) with Spark MLlib or scikit-learn, performs grid search per params.yaml, 
# logs metrics/artifacts to MLflow, and persists model to artifacts/

'''
Reads processed Parquet.

Trains LogisticRegression on features / label.

Saves Spark MLlib model dir (not a .pkl).

'''
from pathlib import Path
from utils.logging import get_logger
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

logger = get_logger("Training")

PROC_PARQUET = "data/processed/train_processed.parquet"
MODEL_DIR = "models/model"

def train_model(in_parquet: str = PROC_PARQUET, model_out_dir: str = MODEL_DIR):
    try:
        logger.info("[LOG] Starting training")

        spark = (SparkSession.builder
                 .appName("TitanicTraining")
                 .getOrCreate())

        df = spark.read.parquet(in_parquet)
        logger.info(f"[LOG] Loaded processed Parquet: rows={df.count()}")

        train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
        model = lr.fit(train_df)
        logger.info("[LOG] Model trained")

        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="label",
            metricName="areaUnderROC"
        )
        val_auc = evaluator.evaluate(model.transform(val_df))
        logger.info(f"[LOG] Validation AUC: {val_auc:.4f}")

        Path(model_out_dir).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_out_dir)
        logger.info(f"[LOG] Model saved -> {model_out_dir}")

        spark.stop()
        logger.info("[LOG] Training completed.")
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    train_model()
