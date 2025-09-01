# Trains classifier(s) with Spark MLlib or scikit-learn, performs grid search per params.yaml, 
# logs metrics/artifacts to MLflow, and persists model to artifacts/

'''
Reads processed Parquet.

Trains LogisticRegression on features / label.

Saves Spark MLlib model dir.

Log hyperparameters.

Log validation AUC.

Log Spark MLlib model as MLflow model.

Register model into MLflow Model Registry.

'''
import mlflow
import mlflow.spark
from pathlib import Path
from utils.logging import get_logger
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import time
import uuid

logger = get_logger("Training")

PROC_PARQUET = "data/processed/train_processed.parquet"
MODEL_DIR = "models/model"
MLFLOW_EXPERIMENT = "Titanic_MLOps"
APP_NAME = "TitanicTraining"

def train_model(in_parquet: str = PROC_PARQUET, model_out_dir: str = MODEL_DIR):
    try:
        logger.info("[LOG] Starting training with MLflow logging")

        

        spark = (
                    SparkSession.builder
                    .appName(APP_NAME)
                    .config("spark.executor.memory", "4g")
                    .config("spark.executor.cores", "2")
                    .config("spark.cores.max", "4")
                    .config("spark.driver.memory", "2g")
                    .config("spark.sql.shuffle.partitions", "8")
                    .getOrCreate()
                )

        # Set MLflow experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="LogReg_Train"):
            df = spark.read.parquet(in_parquet)
            logger.info(f"[LOG] Loaded processed data: {df.count()} rows")

            train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)

            # Model hyperparameters
            max_iter = 50

            # Train
            lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=max_iter)
            model = lr.fit(train_df)
            logger.info("[LOG] Model trained")

            # Evaluate
            evaluator = BinaryClassificationEvaluator(
                rawPredictionCol="rawPrediction",
                labelCol="label",
                metricName="areaUnderROC"
            )
            val_auc = evaluator.evaluate(model.transform(val_df))
            logger.info(f"[LOG] Validation AUC: {val_auc:.4f}")

            # MLflow logging
            mlflow.log_param("maxIter", max_iter)
            mlflow.log_metric("val_auc", val_auc)

            # Save Spark model locally
            Path(model_out_dir).parent.mkdir(parents=True, exist_ok=True)
            model.save(model_out_dir)
            logger.info(f"[LOG] Model saved locally -> {model_out_dir}")

            # Log Spark model to MLflow
            mlflow.spark.log_model(
                                    model,
                                    artifact_path="model",
                                    registered_model_name="TitanicModel"
                                    input_example= train_df.limit(5).toPandas()  
                                )
            logger.info("[LOG] Model logged to MLflow Registry under name 'TitanicModel'")

        spark.stop()
        logger.info("[LOG] Training completed with MLflow logging")

    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    train_model()
