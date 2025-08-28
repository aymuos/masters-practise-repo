from pathlib import Path
from utils.logging import get_logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

logger = get_logger("DataPreprocessing")

RAW_CSV = "data/raw/train.csv"
PROC_PARQUET = "data/processed/train_processed.parquet"
PREPROC_MODEL_DIR = "models/preprocess_pipeline"

def preprocess_titanic(raw_csv: str = RAW_CSV,
                       out_parquet: str = PROC_PARQUET,
                       pipeline_out_dir: str = PREPROC_MODEL_DIR):
    try:
        logger.info("[LOG] Starting Spark preprocessing")

        spark = (SparkSession.builder
                 .appName("TitanicPreprocessing")
                 .getOrCreate())

        # 1) Load raw CSV
        df = spark.read.csv(raw_csv, header=True, inferSchema=True)
        logger.info(f"[LOG] Loaded raw data: rows={df.count()}, cols={len(df.columns)}")

        # 2) Basic cleanup
        drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in df.columns]
        if drop_cols:
            df = df.drop(*drop_cols)
            logger.info(f"[LOG] Dropped columns: {drop_cols}")

        # 3) Impute simple missing values (Age median, Embarked mode)
        # Age median
        if "Age" in df.columns:
            median_age = df.approxQuantile("Age", [0.5], 0.0)[0]
            df = df.fillna({"Age": median_age})
            logger.info("[LOG] Filled missing Age with median")
        # Embarked mode
        if "Embarked" in df.columns:
            mode_embarked = df.groupBy("Embarked").count().orderBy("count", ascending=False).first()[0]
            df = df.fillna({"Embarked": mode_embarked})
            logger.info("[LOG] Filled missing Embarked with mode")

        # 4) Label column ensure name 'label'
        label_col = "Survived"
        if label_col not in df.columns:
            raise ValueError("Expected 'Survived' column in Titanic dataset.")
        df = df.withColumnRenamed(label_col, "label")

        # 5) Categorical + numeric
        cat_cols = [c for c in ["Sex", "Embarked", "Pclass"] if c in df.columns]
        num_cols = [c for c in df.columns if c not in ["label"] + cat_cols + ["PassengerId"]]

        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                    for c in cat_cols]
        encoders = OneHotEncoder(
            inputCols=[f"{c}_idx" for c in cat_cols],
            outputCols=[f"{c}_vec" for c in cat_cols],
            handleInvalid="keep"
        )

        feature_cols = num_cols + [f"{c}_vec" for c in cat_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

        pipe = Pipeline(stages=indexers + [encoders, assembler])

        # 6) Fit preprocessing pipeline and transform
        preproc_model = pipe.fit(df)
        df_proc = preproc_model.transform(df).select("features", col("label").cast("double"))

        # 7) Save outputs
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        df_proc.write.mode("overwrite").parquet(out_parquet)
        logger.info(f"[LOG] Saved processed Parquet -> {out_parquet}")

        Path(pipeline_out_dir).parent.mkdir(parents=True, exist_ok=True)
        preproc_model.save(pipeline_out_dir)
        logger.info(f"[LOG] Saved fitted preprocessing pipeline -> {pipeline_out_dir}")

        spark.stop()
        logger.info("[LOG] Preprocessing completed.")
    except Exception as e:
        logger.error(f"[ERROR] Preprocessing failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    preprocess_titanic()
