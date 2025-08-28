# FastAPI app with /health and /predict endpoints, input schema validation, model loading on startup, and optional probability outputs

'''
Loads preprocessing PipelineModel + production model.

Accepts raw Titanic-like JSON; applies preprocessing → predict.

Returns predicted class + probability.
FastAPI app with /health and /predict endpoints, input schema validation, model loading on startup, and optional probability outputs.


'''

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from utils.logging import get_logger
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

logger = get_logger("API")

PREPROC_MODEL_DIR = "models/preprocess_pipeline"
PROD_MODEL_DIR = "models/production"

app = FastAPI(title="Titanic MLOps API")

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    # You can include more raw fields; the preprocessing pipeline will handle them if configured.

def _load_spark_and_models():
    spark = (SparkSession.builder
             .appName("TitanicServing")
             .getOrCreate())
    preproc = PipelineModel.load(PREPROC_MODEL_DIR)
    model = LogisticRegressionModel.load(PROD_MODEL_DIR)
    return spark, preproc, model

spark, preproc_model, prod_model = None, None, None

@app.on_event("startup")
def startup_event():
    global spark, preproc_model, prod_model
    try:
        logger.info("[LOG] Starting API, loading Spark and models")
        spark, preproc_model, prod_model = _load_spark_and_models()
        logger.info("[LOG] Models loaded successfully")
    except Exception as e:
        logger.error(f"[ERROR] API startup failed: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
def shutdown_event():
    global spark
    try:
        if spark:
            spark.stop()
            logger.info("[LOG] Spark session stopped")
    except Exception as e:
        logger.error(f"[ERROR] API shutdown error: {e}", exc_info=True)

@app.post("/predict")
def predict(passenger: Passenger):
    try:
        # Create a single-row Spark DF
        df = spark.createDataFrame([passenger.dict()])
        # Apply preprocessing pipeline (produces 'features' + 'label' may be absent)
        df_proc = preproc_model.transform(df)
        # Predict
        pred = prod_model.transform(df_proc).select("prediction", "probability").collect()[0]
        prediction = int(pred["prediction"])
        prob = float(pred["probability"][1])  # probability of class 1
        logger.info(f"[LOG] Served prediction={prediction}, prob_1={prob:.4f}")
        return {"prediction": prediction, "probability_1": prob}
    except Exception as e:
        logger.error(f"[ERROR] Inference failed: {e}", exc_info=True)
        return {"error": "Inference failed."}

if __name__ == "__main__":
    # Run with: python deployment/api.py  (or use uvicorn directly)
    uvicorn.run(app, host="0.0.0.0", port=8000)
