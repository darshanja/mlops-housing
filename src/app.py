from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

app = FastAPI()
model = joblib.load("models/best_model.pkl")
logging.basicConfig(filename="logs/api.log", level=logging.INFO)

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

prediction_count = 0

@app.post("/predict")
def predict(features: HouseFeatures):
    global prediction_count
    prediction_count += 1

    data = pd.DataFrame([features.dict()])
    prediction = model.predict(data)[0]

    logging.info(f"Input: {features.dict()}, Prediction: {prediction}")
    return {"prediction": round(float(prediction), 2)}

@app.get("/metrics")
def metrics():
    return f"predictions_total {prediction_count}"
