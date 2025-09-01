from pathlib import Path
from utils.logging import get_logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, isnull, count, sum as spark_sum, mean, stddev, regexp_extract
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
import json

logger = get_logger("DataPreprocessing")

RAW_CSV = "data/raw/train.csv"
PROC_PARQUET = "data/processed/train_processed.parquet"
PREPROC_MODEL_DIR = "models/preprocess_pipeline"
VALIDATION_REPORT_DIR = "reports/data_validation"

# Expected schema for Titanic dataset
EXPECTED_SCHEMA = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Survived", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Ticket", IntegerType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Cabin", StringType(), True),   # alphanumeric string
    StructField("Embarked", StringType(), True)
])

def validate_data_schema(df, expected_schema):
    """Validate that the dataframe matches expected schema"""
    try:
        actual_schema = df.schema

        if actual_schema != expected_schema:
            logger.warning(f"[WARNING] Schema mismatch. \
                            Expected: {expected_schema}, Got: {actual_schema}")
            return False
        logger.info("[LOG]4- Schema validation passed")
        return True
    
    except Exception as e:
        logger.error(f"[ERROR] Schema validation failed: {e}")
        return False

        # _____________________________________________________________________

def validate_data_quality(df):
    """Perform comprehensive data quality checks"""
    logger.info("[LOG] 5- Starting data quality validation")
    
    # creating a quality_report dict that will store information about the data 
    quality_report = {
        "total_rows": df.count(),
        "total_columns": len(df.columns),
        "missing_values": {},
        "data_types": {},
        "value_ranges": {},
        "unique_values": {},
        "quality_issues": []
    }

    # remove columns that are not needed for validation
    columns_to_drop = ["PassengerId","Name", "Ticket", "Cabin"]

    df = df.drop(*columns_to_drop)    # * is the unpacking operator here , it unpacks the list into individual arguments

    logger.info(f"Dropped columns: {columns_to_drop}")
                 
    
    # Check for missing values
    for col_name in df.columns:

        missing_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
        missing_pct = (missing_count / quality_report["total_rows"]) * 100

        quality_report["missing_values"][col_name] = {
            "count": missing_count,
            "percentage": round(missing_pct, 2)
        }
        
        # Log high missing value percentages
        if missing_pct > 50:
            quality_report["quality_issues"].append(f"High missing values in {col_name}: {missing_pct}%")
            logger.warning(f"[WARNING] High missing values in {col_name}: {missing_pct}%")
    

    # Validate specific columns
    # Age validation

    if "Age" in df.columns:

        age_stats = df.select("Age").summary("count", "min", "max", "mean", "stddev").collect()

        quality_report["value_ranges"]["Age"] = {
            "min": age_stats[1]["Age"],
            "max": age_stats[2]["Age"],
            "mean": age_stats[3]["Age"],
            "stddev": age_stats[4]["Age"]
        }
        
        # Check for unrealistic age values should lie between 0 and 100 
        unrealistic_age = df.filter((col("Age") < 0) | (col("Age") > 100)).count()

        if unrealistic_age > 0:
            quality_report["quality_issues"].append(f"Unrealistic age values: {unrealistic_age} rows")
            logger.warning(f"[WARNING] Found {unrealistic_age} rows with unrealistic age values")

            # Remove unrealistic age values
            df = df.filter((col("Age") > 0) & (col("Age") < 100))
            logger.info(f"[LOG] Removed {unrealistic_age} rows with unrealistic age values")

    
    # Fare validation
    if "Fare" in df.columns:
        fare_stats = df.select("Fare").summary("count", "min", "max", "mean", "stddev").collect()

        quality_report["value_ranges"]["Fare"] = {
            "min": fare_stats[1]["Fare"],
            "max": fare_stats[2]["Fare"],
            "mean": fare_stats[3]["Fare"],
            "stddev": fare_stats[4]["Fare"]
        }
        
        # Check for negative or extremely high fares
        invalid_fare = df.filter((col("Fare") < 0.00) | (col("Fare") > 1000.00)).count()

        if invalid_fare > 0:
            quality_report["quality_issues"].append(f"Invalid fare values: {invalid_fare} rows")
            logger.warning(f"[WARNING] Found {invalid_fare} rows with invalid fare values")
    
    # Categorical value validation
    categorical_cols = ["Sex", "Embarked", "Pclass"]
    for cat_col in categorical_cols:
        if cat_col in df.columns:
            unique_vals = df.select(cat_col).distinct().collect()
            quality_report["unique_values"][cat_col] = [row[cat_col] for row in unique_vals]
            
            # Check for unexpected values
            if cat_col == "Sex" and not all(val in ["male", "female"] for val in quality_report["unique_values"][cat_col] if val):
                quality_report["quality_issues"].append(f"Unexpected values in {cat_col}: {quality_report['unique_values'][cat_col]}")
                logger.warning(f"[WARNING] Unexpected values in {cat_col}: {quality_report['unique_values'][cat_col]}")
    
    # Save quality report
    Path(VALIDATION_REPORT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{VALIDATION_REPORT_DIR}/data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)
    
    logger.info(f"[LOG] Data quality report saved to {VALIDATION_REPORT_DIR}/data_quality_report.json")
    
    # Log summary
    total_issues = len(quality_report["quality_issues"])
    if total_issues == 0:
        logger.info("[LOG] No data quality issues found")
    else:
        logger.warning(f"[WARNING] Found {total_issues} data quality issues")
        for issue in quality_report["quality_issues"]:
            logger.warning(f"[WARNING] {issue}")
    
    return quality_report

def advanced_imputation(df):
    """Perform advanced imputation using multiple strategies"""
    logger.info("[LOG] Starting advanced imputation")
    
    # Create a copy for imputation
    df_imputed = df
    
    # 1. Age imputation using median by Sex and Pclass
    if "Age" in df.columns:
        logger.info("[LOG] Imputing Age using median by Sex and Pclass")
        
        # Ensure Age column is numeric
        df_imputed = df_imputed.withColumn("Age", col("Age").cast("double"))
        
        # Calculate median age by Sex and Pclass
        age_medians = df_imputed.groupBy("Sex", "Pclass").agg(mean("Age").alias("median_age"))
        
        # Join with original dataframe and fill missing ages
        df_imputed = df_imputed.join(age_medians, ["Sex", "Pclass"], "left")
        df_imputed = df_imputed.withColumn(
            "Age", 
            when(col("Age").isNull() | isnan(col("Age")), col("median_age")).otherwise(col("Age"))
        )
        df_imputed = df_imputed.drop("median_age")
        
        # If still missing values, use overall median
        overall_median = df_imputed.select("Age").summary("50%").collect()[0]["Age"]
        df_imputed = df_imputed.withColumn(
            "Age",
            when(col("Age").isNull() | isnan(col("Age")), overall_median).otherwise(col("Age"))
        )
        
        logger.info(f"[LOG] Age imputation completed. Overall median: {overall_median}")
    
    # 2. Fare imputation using median by Pclass
    if "Fare" in df.columns:
        logger.info("[LOG] Imputing Fare using median by Pclass")
        
        # Ensure Fare column is numeric
        df_imputed = df_imputed.withColumn("Fare", col("Fare").cast("double"))
        
        # Calculate median fare by Pclass
        fare_medians = df_imputed.groupBy("Pclass").agg(mean("Fare").alias("median_fare"))
        
        # Join and fill missing fares
        df_imputed = df_imputed.join(fare_medians, ["Pclass"], "left")
        df_imputed = df_imputed.withColumn(
            "Fare",
            when(col("Fare").isNull() | isnan(col("Fare")), col("median_fare")).otherwise(col("Fare"))
        )
        df_imputed = df_imputed.drop("median_fare")
        
        logger.info("[LOG] Fare imputation completed")
    
    # 3. Embarked imputation using mode
    if "Embarked" in df.columns:
        logger.info("[LOG] Imputing Embarked using mode")
        
        # Find mode (most frequent value)
        embarked_counts = df_imputed.groupBy("Embarked").count().orderBy("count", ascending=False)
        mode_embarked = embarked_counts.first()[0]
        
        df_imputed = df_imputed.fillna({"Embarked": mode_embarked})
        logger.info(f"[LOG] Embarked imputation completed. Mode: {mode_embarked}")
    
    # 4. Cabin imputation - create a new feature indicating if cabin is known
    if "Cabin" in df.columns:
        logger.info("[LOG] Creating Cabin_known feature")
        
        df_imputed = df_imputed.withColumn(
            "Cabin_known",
            when(col("Cabin").isNull(), 0).otherwise(1)
        )
        
        # Fill missing cabins with "Unknown"
        df_imputed = df_imputed.fillna({"Cabin": "Unknown"})
        logger.info("[LOG] Cabin feature engineering completed")
    
    # 5. Family size feature engineering
    if "SibSp" in df.columns and "Parch" in df.columns:
        logger.info("[LOG] Creating family size features")
        
        df_imputed = df_imputed.withColumn(
            "FamilySize", 
            col("SibSp") + col("Parch") + 1
        )
        
        df_imputed = df_imputed.withColumn(
            "IsAlone",
            when(col("FamilySize") == 1, 1).otherwise(0)
        )
        
        logger.info("[LOG] Family size features created")
    
    # 6. Title extraction from Name
    if "Name" in df.columns:
        logger.info("[LOG] Extracting title from Name")
        
        df_imputed = df_imputed.withColumn(
            "Title",
            regexp_extract(col("Name"), r"([A-Za-z]+)\.", 1)
        )
        
        # Group rare titles
        df_imputed = df_imputed.withColumn(
            "Title",
            when(col("Title").isin(["Mr", "Miss", "Mrs", "Master"]), col("Title"))
            .when(col("Title").isin(["Dr", "Rev", "Col", "Major", "Capt"]), "Officer")
            .when(col("Title").isin(["Jonkheer", "Don", "Sir", "Countess", "Lady"]), "Royalty")
            .otherwise("Other")
        )
        
        logger.info("[LOG] Title extraction completed")
    
    logger.info("[LOG] Advanced imputation completed")
    return df_imputed

def preprocess_titanic(raw_csv: str = RAW_CSV,
                       out_parquet: str = PROC_PARQUET,
                       pipeline_out_dir: str = PREPROC_MODEL_DIR):
    try:
        logger.info("[LOG]1- Starting Spark preprocessing")

        spark = (SparkSession.builder
                 .appName("TitanicPreprocessing")
                 .getOrCreate())
        
        # 1) Load raw CSV
        logger.info("[LOG]2-Loading raw CSV data")
        df = spark.read.csv(raw_csv, header=True, inferSchema=True)
        logger.info(f"[LOG]3- Loaded raw data: rows={df.count()}, cols={len(df.columns)}")


        # 2) Schema validation
        logger.info("[LOG]3- Validating data schema")
        if not validate_data_schema(df, EXPECTED_SCHEMA):
            logger.warning("[WARNING]3a Schema validation failed, but continuing with processing")

        # 3) Data quality validation
        logger.info("[LOG] Performing data quality checks")
        quality_report = validate_data_quality(df)

        # 4) Advanced imputation
        logger.info("[LOG] Performing advanced imputation")
        df_imputed = advanced_imputation(df)

        # 5) Basic cleanup - drop unnecessary columns
        logger.info("[LOG] Cleaning up columns")
        drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in df_imputed.columns]
        if drop_cols:
            df_imputed = df_imputed.drop(*drop_cols)
            logger.info(f"[LOG] Dropped columns: {drop_cols}")

        # 6) Ensure label column name is 'label'
        label_col = "Survived"
        if label_col not in df_imputed.columns:
            raise ValueError("Expected 'Survived' column in Titanic dataset.")
        df_imputed = df_imputed.withColumnRenamed(label_col, "label")

        # 7) Identify categorical and numerical columns
        cat_cols = [c for c in ["Sex", "Embarked", "Pclass", "Title", "Cabin_known", "IsAlone"] 
                    if c in df_imputed.columns]
        num_cols = [c for c in df_imputed.columns 
                    if c not in ["label"] + cat_cols + ["PassengerId"]]

        logger.info(f"[LOG] Categorical columns: {cat_cols}")
        logger.info(f"[LOG] Numerical columns: {num_cols}")

        # 8) Create simplified preprocessing pipeline using only StringIndexer
        logger.info("[LOG] Creating simplified preprocessing pipeline")
        
        # Simple label encoding for categorical columns (no one-hot encoding)
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                    for c in cat_cols]
        
        # Feature assembler - include both numerical and indexed categorical columns
        feature_cols = num_cols + [f"{c}_idx" for c in cat_cols]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

        # Create simplified pipeline
        pipe = Pipeline(stages=indexers + [assembler])

        # 9) Fit preprocessing pipeline and transform
        logger.info("[LOG] Fitting and applying preprocessing pipeline")
        preproc_model = pipe.fit(df_imputed)
        df_proc = preproc_model.transform(df_imputed).select("features", col("label").cast("double"))

        # 10) Final validation
        logger.info("[LOG] Final data validation")
        final_count = df_proc.count()
        if final_count == 0:
            raise ValueError("Final processed dataset is empty!")
        
        logger.info(f"[LOG] Final processed dataset: {final_count} rows")

        # 11) Save outputs
        logger.info("[LOG] Saving processed data and pipeline")
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        df_proc.write.mode("overwrite").parquet(out_parquet)
        logger.info(f"[LOG] Saved processed Parquet -> {out_parquet}")

        Path(pipeline_out_dir).parent.mkdir(parents=True, exist_ok=True)
        preproc_model.save(pipeline_out_dir)
        logger.info(f"[LOG] Saved fitted preprocessing pipeline -> {pipeline_out_dir}")

        # 12) Save processing summary
        processing_summary = {
            "input_rows": df.count(),
            "output_rows": final_count,
            "input_columns": len(df.columns),
            "output_columns": len(df_proc.columns),
            "categorical_features": len(cat_cols),
            "numerical_features": len(num_cols),
            "quality_issues_found": len(quality_report.get("quality_issues", [])),
            "imputation_applied": True,
            "encoding_strategy": "label_encoding"
        }
        
        with open(f"{VALIDATION_REPORT_DIR}/processing_summary.json", "w") as f:
            json.dump(processing_summary, f, indent=2)
        
        logger.info(f"[LOG] Processing summary saved to {VALIDATION_REPORT_DIR}/processing_summary.json")

        spark.stop()
        logger.info("[LOG] Enhanced preprocessing completed successfully.")
        
        return df_proc, preproc_model
        
    except Exception as e:
        logger.error(f"[ERROR] Enhanced preprocessing failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    preprocess_titanic()
