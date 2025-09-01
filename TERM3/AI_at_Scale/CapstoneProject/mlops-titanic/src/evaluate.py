# Computes holdout metrics, AUC, F1, calibration/PR curves, and stores confusion matrix plots; decides “best model” and produces a metrics.json summary

'''
Loads model + processed Parquet.

Computes AUC, accuracy, precision, recall, F1, confusion matrix.

Saves metrics.json and confusion_matrix.png.

'''

import mlflow
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.logging import get_logger
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

logger = get_logger("Evaluation")

PROC_PARQUET = "data/processed/train_processed.parquet"
MODEL_DIR = "models/model"
REPORT_DIR = "reports"
MLFLOW_EXPERIMENT = "Titanic_MLOps"

def evaluate_model(in_parquet: str = PROC_PARQUET,
                   model_dir: str = MODEL_DIR,
                   report_dir: str = REPORT_DIR):
    try:
        logger.info("[LOG] Starting evaluation with MLflow logging")

        spark = SparkSession.builder.appName("TitanicEvaluation").getOrCreate()
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="LogReg_Eval"):
            df = spark.read.parquet(in_parquet)
            logger.info(f"[LOG] Loaded processed data: {df.count()} rows")

            model = LogisticRegressionModel.load(model_dir)
            logger.info("[LOG] Model loaded")

            preds = model.transform(df).select("prediction", "label", "rawPrediction", "probability")

            # Metrics
            evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
            auc = evaluator.evaluate(preds)

            rdd = preds.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
            m = MulticlassMetrics(rdd)

            accuracy = m.accuracy
            precision = m.precision(1.0)
            recall = m.recall(1.0)
            f1 = m.fMeasure(1.0)
            cm = np.array(m.confusionMatrix().toArray())

            # Save metrics JSON
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            metrics = {"auc": auc, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            with open(f"{report_dir}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Save confusion matrix image
            cm_path = f"{report_dir}/confusion_matrix.png"
            plt.figure(figsize=(5, 4))
            plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.colorbar()
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, int(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(cm_path)

            # MLflow logging
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(f"{report_dir}/metrics.json")
            mlflow.log_artifact(cm_path)

            logger.info("[LOG] Evaluation results logged to MLflow")

        spark.stop()
        logger.info("[LOG] Evaluation completed")

    except Exception as e:
        logger.error(f"[ERROR] Evaluation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    evaluate_model()
