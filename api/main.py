import joblib as pickle
import os

from fastapi import FastAPI
from helpers.predict import predict

import uvicorn 

model = None
tlidf = None

with open('./model/mental_health_model.pkl', 'rb') as f:
    model = pickle.load(f)


with open('./model/mental_health_tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)


app = FastAPI()

@app.post('/predict')
def return_predictions(data: dict):
    data = data.get('data', [])
    
    predictions = []

    for text in data:
        prediction = predict(model, tfidf, text)
        predictions.append(prediction)

    return { 'data': data, 'predictions': predictions }


@app.get('/')
def health_check():
    return 'API is up and running!'


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)