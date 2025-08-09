import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import yaml
import sys
import logging
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    try:
        # Load dataset
        logger.info("Loading dataset...")
        df = pd.read_csv("data/housing.csv")
        
        # Ensure all columns are numeric
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            
        # Drop any rows with missing values
        df = df.dropna()
        
        X = df.drop("Price", axis=1)
        y = df["Price"]
        logger.info(f"Training with {len(df)} samples")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Load hyperparameters
        logger.info("Loading hyperparameters...")
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        depth = params["model"]["max_depth"]

        # Setup MLflow
        logger.info("Setting up MLflow...")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("HousingExperiment")

        best_model = None
        best_score = float("inf")
        acceptable_mse_threshold = params.get("training", {}).get("max_mse", 1e6)

        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(max_depth=depth)
        }

        for name, model in models.items():
            logger.info(f"Training {name}...")
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)

                mlflow.log_param("model_type", name)
                mlflow.log_metric("mse", mse)
                mlflow.sklearn.log_model(model, "model")

                logger.info(f"{name} MSE: {mse:.4f}")

                if mse < best_score:
                    best_score = mse
                    best_model = model

        # Validate model performance
        if best_score > acceptable_mse_threshold:
            logger.error(f"Model performance (MSE: {best_score:.4f}) is worse than threshold ({acceptable_mse_threshold})")
            return 1

        # Save the best model
        logger.info(f"Saving best model (MSE: {best_score:.4f})...")
        joblib.dump(best_model, "models/best_model.pkl")
        return 0

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(train())
