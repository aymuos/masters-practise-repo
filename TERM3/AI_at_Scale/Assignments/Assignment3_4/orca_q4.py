import os
import socket
import time
import argparse

from bigdl.orca import init_orca_context, OrcaContext, stop_orca_context
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy, Precision, Recall
import time


print("SPARK_HOME =", os.environ.get("SPARK_HOME"))
if "SPARK_HOME" in os.environ:
    del os.environ["SPARK_HOME"]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


init_orca_context(cluster_mode="local", cores=4, memory="6g")
sc = OrcaContext.get_spark_context()
start_time = time.time()

print("OrcaContext initialized with Spark context:", sc)
# Simple custom CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10),
            # Optional: Add LogSoftmax if you want to keep NLLLoss
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_cifar10_datasets():
    print("Downloading CIFAR-10 datasets...")
    """Download and create a consistent subset of CIFAR-10 datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_full = datasets.CIFAR10("/tmp/cifar10", train=True, download=True, transform=transform)
    test_full = datasets.CIFAR10("/tmp/cifar10", train=False, download=True, transform=transform)

    print("CIFAR-10 datasets downloaded.")
    
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Create fixed subsets
    train_indices = torch.randperm(len(train_full))[:2000].tolist()
    test_indices = torch.randperm(len(test_full))[:2000].tolist()
    
    train_subset = torch.utils.data.Subset(train_full, train_indices)
    test_subset = torch.utils.data.Subset(test_full, test_indices)

    print(f"Created train subset with {len(train_subset)} samples and test subset with {len(test_subset)} samples.")

    return train_subset, test_subset

# Load the consistent subsets once
train_dataset, test_dataset = get_cifar10_datasets()

def train_loader_creator(config):
    print("Creating train dataloader...")
    # Dataloader is created directly from the pre-loaded dataset
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True)

def test_loader_creator(config):
    print("Creating test dataloader...")
    # Dataloader is created directly from the pre-loaded dataset
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False)


# model
def model_creator(config):
    print("Creating model...")
    # No need to manually move to a device; Orca handles it
    model = SimpleCNN()
    return model

# optimizer
def optimizer_creator(model, config):
    print("Creating optimizer...")
    return torch.optim.Adam(model.parameters(), config["lr"])

# estimator
est = Estimator.from_torch(model=model_creator,
                           optimizer=optimizer_creator,
                           # Use CrossEntropyLoss, as it's compatible with the model's output
                           loss=nn.CrossEntropyLoss(),
                           metrics=[Accuracy(), Precision(), Recall()],
                           use_tqdm=True,
                           config={"lr": 1e-2,
                                   "batch_size": 64
                                   })

print("Estimator created.")
train_stats = est.fit(data=train_loader_creator, epochs=1)
print(train_stats)

print("Training completed. Evaluating on test set...")
# Evaluate the model on the test set
eval_stats = est.evaluate(data=test_loader_creator)
print(eval_stats)
print("TOTAL RUN TIME: ", time.time() - start_time)

est.save("CIFAR_10_Model")
stop_orca_context()