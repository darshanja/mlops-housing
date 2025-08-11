from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from fastapi.responses import Response

app = FastAPI(title="Housing Price Prediction API")

# Initialize the model
model = joblib.load("models/best_model.pkl")
logging.basicConfig(filename="logs/api.log", level=logging.INFO)

# Define Prometheus metrics
PREDICTIONS = Counter('housing_predictions_total', 'Number of predictions made')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Time spent processing prediction')
MODEL_VERSION = Gauge('model_version', 'Model version')

# Initialize model version
MODEL_VERSION.set(1.0)

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
async def predict(features: HouseFeatures):
    try:
        start_time = time.time()
        data = pd.DataFrame([features.dict()])
        prediction = model.predict(data)[0]
        
        # Update metrics
        PREDICTIONS.inc()
        PREDICTION_DURATION.observe(time.time() - start_time)
        
        logging.info(f"Input: {features.dict()}, Prediction: {prediction}")
        return {"prediction": round(float(prediction), 2)}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise
        prediction_latency.observe(time.time() - start_time)

@app.get("/metrics")
def metrics():
    return f"predictions_total {prediction_count}"
