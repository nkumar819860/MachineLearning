from fastapi import FastAPI
import joblib
import numpy as np
import os

app= FastAPI()

MODEL_PATH = os.listdir("saved_models")[0]
model= joblib.load(f"saved_models/{MODEL_PATH}")

@app.get('/')
def health():
    return {"status": "running and healthy"}

@app.post("/predict")
def predict(features: list):
    data= np.array(features).reshape(1,-1)
    return {"prediction": int(model.predict(data)[0])}

