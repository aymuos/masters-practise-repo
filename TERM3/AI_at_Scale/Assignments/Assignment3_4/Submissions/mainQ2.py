import torch
import torchvision
from torchvision import models, datasets, transforms
import numpy as np
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
import random
import os
import sys
import traceback

# ----------------------------------------------------------
# Utility: log and safe-exit
# ----------------------------------------------------------
def log(msg):
    print(f"[LOG] {msg}")

def log_error(msg, e=None):
    print(f"[ERROR] {msg}")
    if e:
        traceback.print_exc(file=sys.stdout)


# ----------------------------------------------------------
# Load MobileNetV2 model
# ----------------------------------------------------------
try:
    log("Loading MobileNetV2 model...")
    mobilenet = models.mobilenet_v2(pretrained=False)
    mobilenet.load_state_dict(
        torch.load('/opt/spark/data/pretrained_models/mobilenet_v2-b0353104.pth')
    )
    log("MobileNetV2 model loaded successfully.")
except Exception as e:
    log_error("Failed to load MobileNetV2 model.", e)
    sys.exit(1)


# ----------------------------------------------------------
# Define transforms
# ----------------------------------------------------------
try:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    log("Image transforms created successfully.")
except Exception as e:
    log_error("Error creating transforms.", e)
    sys.exit(1)


# ----------------------------------------------------------
# Load datasets
# ----------------------------------------------------------
try:
    trainset_fe = datasets.CIFAR10(root='data', train=True, download=False, transform=transform)
    testset_fe = datasets.CIFAR10(root='data', train=False, download=False, transform=transform)
    log("CIFAR-10 datasets loaded successfully.")
except Exception as e:
    log_error("Failed to load CIFAR-10 datasets.", e)
    sys.exit(1)


# ----------------------------------------------------------
# Prepare model for feature extraction
# ----------------------------------------------------------
try:
    mobilenet.classifier = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet = mobilenet.to(device)
    mobilenet.eval()
    log(f"Model moved to device: {device} and set to eval mode.")
except Exception as e:
    log_error("Error preparing model for feature extraction.", e)
    sys.exit(1)


# ----------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------
def extract_random_features(dataset, sample_size, batch_size=32):
    try:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:sample_size]

        features_list, labels_list = [], []
        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_images, batch_labels = [], []
                for idx in batch_indices:
                    img, label = dataset[idx]
                    batch_images.append(img)
                    batch_labels.append(label)

                images_tensor = torch.stack(batch_images).to(device)
                batch_feats = mobilenet(images_tensor).cpu().numpy()

                features_list.extend(batch_feats)
                labels_list.extend(batch_labels)

        log(f"Extracted {len(features_list)} features successfully.")
        return np.array(features_list), np.array(labels_list)

    except Exception as e:
        log_error("Error extracting features.", e)
        return np.array([]), np.array([])


try:
    log("Extracting features from training and test datasets...")
    train_features, train_labels = extract_random_features(trainset_fe, 10000, batch_size=128)
    test_features, test_labels = extract_random_features(testset_fe, 500, batch_size=128)
    log(f"Train features: {train_features.shape}, Test features: {test_features.shape}")
except Exception as e:
    log_error("Feature extraction failed.", e)
    sys.exit(1)


# ----------------------------------------------------------
# Spark session creation
# ----------------------------------------------------------
def create_spark_session(app_name, executor_memory="10G", executor_cores=6, cores_max=6):
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.executor.cores", executor_cores) \
            .config("spark.cores.max", cores_max) \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "200") \
            .getOrCreate()
        log(f"Spark session created: {app_name} "
            f"(executor_memory={executor_memory}, executor_cores={executor_cores}, max_cores={cores_max})")
        return spark
    except Exception as e:
        log_error("Failed to create Spark session.", e)
        sys.exit(1)


# ----------------------------------------------------------
# Convert features/labels to Spark DF
# ----------------------------------------------------------
def to_spark_df(spark, features, labels):
    try:
        rows = [Row(features=Vectors.dense(f.tolist()), label=float(l))
                for f, l in zip(features, labels)]
        df = spark.createDataFrame(rows)
        log(f"Converted numpy arrays to Spark DataFrame with {df.count()} rows.")
        return df
    except Exception as e:
        log_error("Failed to convert features to Spark DataFrame.", e)
        return None


# ----------------------------------------------------------
# Train and evaluate classifiers
# ----------------------------------------------------------
def train_and_evaluate(train_df, test_df):
    results = []
    try:
        log("Starting model training and evaluation...")
        evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
        evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")

        # Logistic Regression
        try:
            log("Training Logistic Regression...")
            start = time.time()
            lr_model = LogisticRegression(maxIter=50, regParam=0.01).fit(train_df)
            lr_preds = lr_model.transform(test_df)
            duration = time.time() - start
            results.append((
                "Logistic Regression",
                evaluator_acc.evaluate(lr_preds),
                evaluator_f1.evaluate(lr_preds),
                duration
            ))
            log("Logistic Regression completed successfully.")
        except Exception as e:
            log_error("Logistic Regression failed.", e)

        # Random Forest
        try:
            log("Training Random Forest...")
            start = time.time()
            rf_model = RandomForestClassifier(
                            numTrees=20,
                            maxDepth=6, 
                            featureSubsetStrategy="sqrt", 
                            subsamplingRate=0.8
            ).fit(train_df)                                                    # SPARK FITS THIS INTO DISTRIBUTED NODES
            
                                                                                # TREES ARE TRAINED IN PARALLEL
            rf_preds = rf_model.transform(test_df)
            duration = time.time() - start

            results.append((
                "Random Forest",
                evaluator_acc.evaluate(rf_preds),
                evaluator_f1.evaluate(rf_preds),
                duration
            ))

            log("Random Forest completed successfully.")
        except Exception as e:
            log_error("Random Forest failed.", e)

        # Naive Bayes
        try:
            log("Training Naive Bayes...")
            start = time.time()
            nb_model = NaiveBayes(smoothing=1.0, modelType="multinomial").fit(train_df)
            nb_preds = nb_model.transform(test_df)
            duration = time.time() - start
            results.append((
                "Naive Bayes",
                evaluator_acc.evaluate(nb_preds),
                evaluator_f1.evaluate(nb_preds),
                duration
            ))
            log("Naive Bayes completed successfully.")
        except Exception as e:
            log_error("Naive Bayes failed.", e)

    except Exception as e:
        log_error("Training and evaluation pipeline failed.", e)

    return results


# ----------------------------------------------------------
# Run experiments with configs
# ----------------------------------------------------------
configs = [
    {"executor_cores": 1, "max_cores": 2, "executor_memory": "4g"},
    {"executor_cores": 2, "max_cores": 4, "executor_memory": "6g"},
    {"executor_cores": 4, "max_cores": 6, "executor_memory": "8g"}
]

all_results = []

if 'spark' in locals() and spark:
    try:
        spark.stop()
        log("Stopped existing Spark session.")
    except Exception:
        pass

for i, cfg in enumerate(configs, start=1):
    try:
        log(f"\n=== Running Configuration {i}: {cfg} ===")
        student_id = "CH24M571"
        app_name = student_id + "_Assignment_3_4"

        spark = create_spark_session(
            app_name=f"{app_name}-{i}",
            executor_cores=cfg["executor_cores"],           # PARALLELISM 2 CORES PER EXECUTOR
            cores_max=cfg["max_cores"],                     # PARALLELISM MAX 4 CORES
            executor_memory=cfg["executor_memory"],
        )

        train_df = to_spark_df(spark, train_features, train_labels).repartition(8)      # DISTRIBUTE DATA INTO 8 PARTITIONS
        test_df = to_spark_df(spark, test_features, test_labels).repartition(8)

        results = train_and_evaluate(train_df, test_df)
        all_results.append((cfg, results))

        spark.stop()
        log("Spark session stopped for this configuration.")

    except Exception as e:
        log_error(f"Pipeline failed for configuration {i}.", e)


# ----------------------------------------------------------
# Print results
# ----------------------------------------------------------
log("\n===== Final Comparison =====")
for cfg, res in all_results:
    print(f"\nConfig: {cfg}")
    print("{:<20} {:<12} {:<12} {:<10}".format("Model", "Accuracy", "F1-score", "Time(s)"))
    for name, acc, f1, t in res:
        print("{:<20} {:<12.4f} {:<12.4f} {:<10.2f}".format(name, acc, f1, t))
