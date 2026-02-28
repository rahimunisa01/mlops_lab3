from fastapi import FastAPI, HTTPException
from data import IrisData, IrisResponse
from predict import predict
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Iris Prediction API is running"}

@app.post("/predict", response_model=IrisResponse)
def make_prediction(data: IrisData):
    try:
        result = predict(data)
        return IrisResponse(prediction=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
