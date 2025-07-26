import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/housing.csv")
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load hyperparameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)
depth = params["model"]["max_depth"]

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("HousingExperiment")

best_model = None
best_score = float("inf")

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=depth)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")

        if mse < best_score:
            best_score = mse
            best_model = model

joblib.dump(best_model, "models/best_model.pkl")
