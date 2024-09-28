import joblib as pickle
import os

from fastapi import FastAPI, Response
from helpers.predict import predict

import uvicorn 
from fastapi.middleware.cors import CORSMiddleware

model = None
tlidf = None

with open('./model/mental_health_model.pkl', 'rb') as f:
    model = pickle.load(f)


with open('./model/mental_health_tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)


app = FastAPI()

# add cors 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Only allow this domain
    allow_methods=["GET", "POST"],  # Allow only GET and POST methods
    allow_headers=["*"],  # Allow all headers
)

@app.post('/predict')
def return_predictions(data: dict):
    data = data.get('data', [])
    
    predictions = []

    for text in data:
        prediction = predict(model, tfidf, text)
        predictions.append(prediction)

    return { 'data': data, 'predictions': predictions }


@app.get('/')
def health_check(response: Response):
    response.status_code = 200
    return 'API is up and running!'


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)