import pytest
import pandas as pd
import os
from src.train import train

def test_data_exists():
    assert os.path.exists("data/housing.csv"), "Housing dataset not found"

def test_data_format():
    df = pd.read_csv("data/housing.csv")
    required_columns = ["Price"]  # Add other required columns
    for col in required_columns:
        assert col in df.columns, f"Required column {col} not found in dataset"
    
def test_training_runs():
    result = train()
    assert result == 0, f"Training failed with return code {result}"
