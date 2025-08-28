# Computes holdout metrics, AUC, F1, calibration/PR curves, and stores confusion matrix plots; decides “best model” and produces a metrics.json summary

'''
Loads model + processed Parquet.

Computes AUC, accuracy, precision, recall, F1, confusion matrix.

Saves metrics.json and confusion_matrix.png.

'''

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

def evaluate_model(in_parquet: str = PROC_PARQUET,
                   model_dir: str = MODEL_DIR,
                   report_dir: str = REPORT_DIR):
    try:
        logger.info("[LOG] Starting evaluation")

        spark = (SparkSession.builder
                 .appName("TitanicEvaluation")
                 .getOrCreate())

        df = spark.read.parquet(in_parquet)
        logger.info(f"[LOG] Loaded processed data: rows={df.count()}")

        model = LogisticRegressionModel.load(model_dir)
        logger.info("[LOG] Model loaded")

        preds = model.transform(df).select("prediction", "label", "rawPrediction", "probability")
        logger.info("[LOG] Predictions generated")

        evaluator = BinaryClassificationEvaluator(
            rawPredictionCol="rawPrediction",
            labelCol="label",
            metricName="areaUnderROC"
        )
        auc = evaluator.evaluate(preds)

        # Detailed metrics via RDD
        rdd = preds.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
        m = MulticlassMetrics(rdd)
        accuracy = m.accuracy
        precision = m.precision(1.0)
        recall = m.recall(1.0)
        f1 = m.fMeasure(1.0)
        cm = np.array(m.confusionMatrix().toArray())

        # Save metrics
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        metrics = {"auc": auc, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        with open(f"{report_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"[LOG] Metrics saved -> {report_dir}/metrics.json")

        # Save confusion matrix plot
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["0", "1"])
        plt.yticks(tick_marks, ["0", "1"])
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(f"{report_dir}/confusion_matrix.png")
        logger.info(f"[LOG] Confusion matrix saved -> {report_dir}/confusion_matrix.png")

        spark.stop()
        logger.info("[LOG] Evaluation completed.")
    except Exception as e:
        logger.error(f"[ERROR] Evaluation failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    evaluate_model()
