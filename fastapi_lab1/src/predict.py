import joblib
import numpy as np

MODEL_PATH = "../model/iris_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict(data):
    model = load_model()
    features = np.array([
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]).reshape(1, -1)

    prediction = model.predict(features)
    return int(prediction[0])
