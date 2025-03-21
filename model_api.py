from fastapi import FastAPI
import pickle
import numpy as np
import yaml

# Load config
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

dataset = config["data_pipelines"]["requested_dataset"]
model_path = f"models/{dataset}/logistic_regression.pkl"

# Load trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict/")
def predict(features: list):
    """Predict using logistic regression model."""
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
