from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = DecisionTreeClassifier()
    model.fit(X, y)

    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, "../model/iris_model.pkl")
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
