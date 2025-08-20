import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models


# Start Spark
student_id = "CH24M571"  # your roll number, letters must be in capital letter
app_name = student_id + "_Assignment_3_4"
assignment_no = "Assignment_3_4_ORCA"

spark = SparkSession.builder \
    .appName(app_name) \
    .config("spark.executor.memory", "6G") \
    .config("spark.executor.cores", "4") \
    .config("spark.cores.max", "8") \
    .config("spark.driver.memory", "10G") \
    .getOrCreate()


# Loading the CIFAR10 dataset
# Define a transform to normalize the data

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])
])

# Load the training and test datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print("#### CIFAR-10 dataset loaded successfully !!!")
