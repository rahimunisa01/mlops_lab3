# FastAPI Iris ML API

## Overview

This project demonstrates how to train a machine learning model and expose it as a REST API using FastAPI.

A Random Forest Classifier is trained on the Iris dataset and served through an API endpoint that allows users to send feature values and receive predicted flower class labels.

---

## Project Structure

    fastapi_lab1/
    ├── assets/
    │   ├── swagger_home.png
    │   ├── predict_request.png
    │   ├── predict_response.png
    │   └── server_running.png
    ├── model/
    │   └── iris_model.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── train.py
    │   ├── predict.py
    │   └── main.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

---

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pydantic
- Uvicorn
- NumPy
- Joblib

---

## Setup Instructions

### 1. Clone the Repository

    git clone <your_repo_url>
    cd fastapi_lab1

### 2. Create Virtual Environment

    python3 -m venv venv
    source venv/bin/activate

### 3. Install Dependencies

    pip install -r requirements.txt

---

## Train the Model

Navigate to the `src` directory and run:

    cd src
    python train.py

This will train a Random Forest model on the Iris dataset and save it to:

    model/iris_model.pkl

---

## Run the API

From the `src` directory:

    uvicorn main:app --reload

Open your browser and go to:

    http://127.0.0.1:8000/docs

This will open the automatically generated Swagger UI.

---

## API Endpoints

### GET /

Returns a welcome message confirming the API is running.

---

### GET /health

Returns API health status.

Example response:

    {
      "status": "healthy"
    }

---

### POST /predict

Predicts the Iris flower class based on input features.

Example Request:

    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }

Example Response:

    {
      "prediction": 0
    }

---

## Model Details

- Dataset: Iris Dataset  
- Algorithm: Random Forest Classifier  
- Output: Flower class (0, 1, or 2)  
- Model loads once at application startup for improved performance  

---

## Screenshots
<img width="1384" height="790" alt="Screenshot 2026-03-03 at 11 28 16 AM" src="https://github.com/user-attachments/assets/683d76d2-ac62-460e-8bec-d6bb43df7cd8" />



---

## Notes

- The model file (`iris_model.pkl`) is excluded from version control.
- The virtual environment (`venv/`) is excluded via `.gitignore`.
- The API uses Pydantic models for input validation.
- If invalid input is provided, FastAPI returns a 422 validation error.
