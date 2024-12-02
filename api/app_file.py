from fastapi import FastAPI
import pickle
from model.model import forecast

app = FastAPI()

# define root
# first endpoint
@app.get("/")
def status():
    return {"API": "connected"}

@app.get("/predict")
def predict(X):

    #model = pickle.load_model()
    prediction = "test"

    return {'forecast': prediction}
