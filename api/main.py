import joblib as pickle

from fastapi import FastAPI
from helpers.predict import predict

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



# uvicorn main:app 
# -> No open ports detected on 0.0.0.0, continuing to scan...

# define a port to run the app
# uvicorn main:app --port 8000

# define a host and port to run the app
# uvicorn main:app --host