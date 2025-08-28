THe plan !!

1. setting up this repo with structure and DVC 
2. use titanic dataset to ensure we can load data clean and split comfortably 
3. Train a simple classifier (LogReg / RandomForest in Spark MLlib).
4. Log all experiments (params, metrics, confusion matrix) to MLflow Tracking.
5. Register the best model in MLflow Registry.
6. Export best model → serve with FastAPI (simpler + faster than Flask).
7. Containerize with Docker (basic Dockerfile).
8. Write a quick test script to hit the API.
9. Auto retraining: write a Python script that simulates new data, re-runs pipeline (call training script), and re-registers model.
10. 





Commands : 

> Inside of mlops-titanic execute ```python -m venv .venv-mlops``` to create the virtual env 
> Execute ``` source .venv-mlops/bin/activate ``` 


# DVC 

git is initialised at root , so dvc shall also be initialised at root 

