# Computes distribution shift (e.g., PSI, KS) between reference and incoming data, writes a drift.json with per-feature stats,
# and exits non-zero or logs an alert when thresholds in params.yaml are exceeded

'''
Compares reference Parquet vs incoming batch Parquet using PSI per feature.

Flags drift if any PSI > threshold (e.g., 0.2).

Saves reports/drift_report.json.

'''


import json
from pathlib import Path
import numpy as np

from utils.logging import get_logger
from pyspark.sql import SparkSession

logger = get_logger("DriftDetection")

REF_PARQUET = "data/processed/train_processed.parquet"       # reference features+label
INCOMING_PARQUET = "data/incoming/batch.parquet"              # same schema: features,label (or at least features)
REPORT_PATH = "reports/drift_report.json"
PSI_THRESHOLD = 0.2
BINS = 10

def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index between two binned distributions."""
    expected = expected / (expected.sum() + eps)
    actual = actual / (actual.sum() + eps)
    return float(np.sum((actual - expected) * np.log((actual + eps) / (expected + eps))))

def _vector_to_array_col(spark_df, colname="features"):
    # Convert Spark Dense/Sparse Vector to array for per-feature analysis
    from pyspark.ml.functions import vector_to_array
    return spark_df.withColumn("features_arr", vector_to_array(colname))

def detect_drift(ref_parquet: str = REF_PARQUET,
                 incoming_parquet: str = INCOMING_PARQUET,
                 report_path: str = REPORT_PATH,
                 psi_threshold: float = PSI_THRESHOLD):
    try:
        logger.info("[LOG] Starting drift detection")

        spark = (SparkSession.builder
                 .appName("TitanicDriftDetection")
                 .getOrCreate())

        ref = spark.read.parquet(ref_parquet)
        inc = spark.read.parquet(incoming_parquet)
        logger.info(f"[LOG] Loaded reference rows={ref.count()} incoming rows={inc.count()}")

        ref = _vector_to_array_col(ref, "features")
        inc = _vector_to_array_col(inc, "features")

        # Collect small samples to driver for PSI (Titanic is tiny; OK for demo)
        ref_arr = np.array(ref.select("features_arr").toPandas()["features_arr"].tolist())
        inc_arr = np.array(inc.select("features_arr").toPandas()["features_arr"].tolist())

        if ref_arr.size == 0 or inc_arr.size == 0:
            raise ValueError("Empty arrays for PSI computation.")

        n_features = ref_arr.shape[1]
        psi_per_feature = {}
        drift_flags = {}

        for j in range(n_features):
            ref_col = ref_arr[:, j]
            inc_col = inc_arr[:, j]
            # Bin by reference deciles to stabilize PSI; fallback to linspace
            try:
                bins = np.unique(np.quantile(ref_col, q=np.linspace(0, 1, BINS + 1)))
                if bins.size < 3:  # degenerate
                    bins = np.linspace(ref_col.min(), ref_col.max() + 1e-9, BINS + 1)
            except Exception:
                bins = np.linspace(ref_col.min(), ref_col.max() + 1e-9, BINS + 1)

            ref_hist, _ = np.histogram(ref_col, bins=bins)
            inc_hist, _ = np.histogram(inc_col, bins=bins)
            psi = _psi(ref_hist.astype(float), inc_hist.astype(float))
            psi_per_feature[f"f{j}"] = psi
            drift_flags[f"f{j}"] = bool(psi > psi_threshold)

        overall_drift = any(drift_flags.values())

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({
                "psi_threshold": psi_threshold,
                "overall_drift": overall_drift,
                "psi_per_feature": psi_per_feature,
                "drift_flags": drift_flags
            }, f, indent=2)
        logger.info(f"[LOG] Drift report saved -> {report_path} | overall_drift={overall_drift}")

        spark.stop()
        logger.info("[LOG] Drift detection completed.")
    except Exception as e:
        logger.error(f"[ERROR] Drift detection failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    detect_drift()

