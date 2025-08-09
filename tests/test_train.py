import pytest
import pandas as pd
import os
import shutil
from src.train import train

@pytest.fixture(scope="session")
def test_data():
    """Create test environment with sample data."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Copy test data to the expected location
    shutil.copy("tests/test_data/test_housing.csv", "data/housing.csv")
    
    yield
    
    # Cleanup after tests
    try:
        os.remove("data/housing.csv")
    except:
        pass

def test_data_exists(test_data):
    """Test if the dataset file exists."""
    assert os.path.exists("data/housing.csv"), "Housing dataset not found"

def test_data_format(test_data):
    """Test if the dataset has the required columns."""
    df = pd.read_csv("data/housing.csv")
    required_columns = ["Price", "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                       "Population", "AveOccup", "Latitude", "Longitude"]
    for col in required_columns:
        assert col in df.columns, f"Required column {col} not found in dataset"
    
def test_training_runs(test_data):
    """Test if the training pipeline runs successfully."""
    result = train()
    assert result == 0, f"Training failed with return code {result}"
