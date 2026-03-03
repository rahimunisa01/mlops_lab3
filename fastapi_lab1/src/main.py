from fastapi import FastAPI, HTTPException
from data import IrisData, IrisResponse
import joblib
import numpy as np
import os

app = FastAPI()

MODEL_PATH = "../model/iris_model.pkl"
model = None  # global model variable


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found. Train the model first.")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")


@app.get("/")
def root():
    return {"message": "Iris Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=IrisResponse)
def make_prediction(data: IrisData):
    try:
        features = np.array([
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]).reshape(1, -1)

        prediction = model.predict(features)
        return IrisResponse(prediction=int(prediction[0]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))