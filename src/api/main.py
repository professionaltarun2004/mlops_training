from fastapi import FastAPI
from src.api.predictor import Predictor
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse
)
import pandas as pd  


app=FastAPI()
predictor=Predictor()

@app.post(
    "/predict",
    response_model=PredictionResponse
)

def predict(request:PredictionRequest):
    prediction=predictor.predict(
        request.text
    )

    return PredictionResponse(
        prediction=prediction
    )

@app.get('/metrics')
def metrics():
    try:
        df=pd.read_csv("logs/predictions.csv")
        return {
            "total_predictions":len(df),
            "prediction_counts":df["prediction"].value_counts().to_dict()
        }
    
    except Exception:
        return {
            "total_predictions":0,
            "prediction_counts":{}
        }

@app.get('/')

def home():
    return {"message": "Welcome to MLOPs Training"}

@app.get("/health")
def health():
    return {"status":"healthy"}

@app.get("/model-info")
def model_info():
    return {
        "model_name":"TextClassifier",
        "alias":"production"
    }
