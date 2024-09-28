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

# expected json body:
# {
#     "data": ["I am feeling very sad", "I am feeling very happy"]
# }

# end.
# run the API using the command: uvicorn main:app --reload
# The API will be hosted on http://localhost:8000