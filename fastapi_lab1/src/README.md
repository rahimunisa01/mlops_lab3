# FastAPI Iris ML Model

This project trains a Decision Tree model on the Iris dataset and exposes it as a FastAPI API.

## Setup

1. Create virtual environment
2. Install requirements:
   pip install -r requirements.txt

## Train Model

cd src
python train.py

## Run API

uvicorn main:app --reload

Open:
http://127.0.0.1:8000/docs
