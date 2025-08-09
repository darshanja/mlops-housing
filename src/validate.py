import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import logging
import sys
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate():
    try:
        # Load the model
        logger.info("Loading trained model...")
        model = joblib.load("models/best_model.pkl")

        # Load validation data
        logger.info("Loading validation data...")
        df = pd.read_csv("data/housing.csv")
        X = df.drop("Price", axis=1)
        y = df["Price"]

        # Load parameters
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        validation_params = params.get("validation", {})
        min_r2_score = validation_params.get("min_r2_score", 0.0)
        max_mse = validation_params.get("max_mse", 30)

        # Make predictions
        logger.info("Making predictions...")
        predictions = model.predict(X)

        # Calculate metrics
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)

        logger.info(f"Validation Metrics:")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"MSE: {mse:.4f}")

        # Check if model meets validation criteria
        failed = False
        if r2 < min_r2_score:
            logger.warning(f"Model R² score ({r2:.4f}) is below threshold ({min_r2_score})")
            failed = True
            
        if mse > max_mse:
            logger.error(f"Model MSE ({mse:.4f}) is above threshold ({max_mse})")
            failed = True

        if failed:
            return 1

        logger.info("Model validation successful!")
        return 0

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(validate())
