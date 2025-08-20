import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torchvision import models
from torchvision import datasets, transforms
import numpy as np

from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

