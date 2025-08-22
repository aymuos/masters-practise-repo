# orca_q4_fixed.py
import os, time, argparse, torch, torch.nn as nn
from torchvision import datasets, transforms
from bigdl.orca import init_orca_context, OrcaContext, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy


import sys, os
print("Driver Python:", sys.executable)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
print("PYSPARK_PYTHON set to:", os.environ["PYSPARK_PYTHON"])


# --------- Model (with justification) ----------
# Two conv blocks (32/64) + BatchNorm + Dropout -> stable training, regularization.
# GlobalAvgPool -> drastically fewer params vs big Linear on 8x8x64.
# Final Linear(64->10). CrossEntropyLoss expects logits (no softmax).


class SmallCIFARNet(nn.Module):
    def __init__(self, p_drop=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),          # 32x16x16
            nn.Dropout(p_drop),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),          # 64x8x8
            nn.Dropout(p_drop),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 64x1x1 (global average pooling)
            nn.Flatten(),             # 64
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def make_loader_creator(subset_size, train=True, num_workers=2):
    # Build datasets inside the creator so each worker constructs them correctly.
    def creator(config,batch_size):
        tfm = transforms.Compose([
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616))
        ])
        ds = datasets.CIFAR10("/tmp/cifar10", train=train, download=True, transform=tfm)
        torch.manual_seed(42)  # reproducible subset
        idx = torch.randperm(len(ds))[:subset_size].tolist()
        sub = torch.utils.data.Subset(ds, idx)
        return torch.utils.data.DataLoader(
            sub, batch_size=batch_size ,shuffle=train,
            num_workers=num_workers, pin_memory=True)
    return creator

def model_creator(_): 
    return SmallCIFARNet(p_drop=0.25)


def optim_creator(model, cfg): 
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"])

def run_once(cores, subset, epochs=5):
    ctx = init_orca_context(
        cluster_mode="local",
        cores=cores,
        memory="6g",
        extra_spark_conf={
            "spark.default.parallelism": str(max(cores*2, 2)),
            "spark.sql.shuffle.partitions": str(max(cores*2, 2)),
        }
    )
    # optional: print(OrcaContext.get_spark_context())
    est = Estimator.from_torch(
    model=model_creator,
    optimizer=optim_creator,
    loss=nn.CrossEntropyLoss(),
    metrics=[Accuracy()],
    use_tqdm=True,
    config={"lr": 1e-3} # learning rate
    )
    
    train_c = make_loader_creator(subset_size=subset, train=True)
    test_c  = make_loader_creator(subset_size=int(subset*0.2), train=False)  # small held-out test

    train_stats = est.fit(data=train_c, epochs=epochs,batch_size=128)

    eval_stats  = est.evaluate(data=test_c,
                              batch_size=128)
    print("Train stats:", train_stats)
    print("Eval stats:", eval_stats)
    est.shutdown()  # free PyTorch workers before reconfiguring Spark
    stop_orca_context()
    return {"cores": cores, "subset": subset, "train": train_stats, "eval": eval_stats}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # ---- Q1/Q2: same dataset & subset size = 10,000 images total (train split) ----
    
    # 10,000 for training; 2,000 test subset for evaluation.
    results = []
    for cores in (2, 4, 8):
        results.append(run_once(cores=cores, subset=10000, epochs=args.epochs))

    # pick the best by Accuracy
    best = max(results, key=lambda r: r["eval"].get("Accuracy", 0.0))



    print("Q2 — parallelism comparison:", results)
    print("Best Spark config:", best["cores"], "cores")
    

    # ---- Q3: retrain/evaluate using 20,000 images with best Spark config ----
    final = run_once(cores=2, subset=20_000, epochs=args.epochs)
    print("Q3 — 20k images result:", final)
