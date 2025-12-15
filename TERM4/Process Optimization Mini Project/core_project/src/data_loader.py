import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_loop(file_path):
    df = pd.read_csv(file_path)

    # Standardize column names
    df = df.rename(columns={
        "pv_1": "PV",
        "sp_1": "SP",
        "op_1": "OP"
    })


    # Forward-fill missing values
    df = df.ffill().bfill()

    return df
