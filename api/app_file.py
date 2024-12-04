from fastapi import FastAPI
from model.model import forecast
from tensorflow import keras
from Utils import preprocessing
from Utils import matrix_creation
from Utils import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


app = FastAPI()

# define root
# first endpoint
@app.get("/")
def status():
    return {"API": "connected, test francois"}

@app.get("/predict")
def predict(X):

    pgn=preprocessing.extract_moves_chess(X)
    X = matrix_creation.create_matrice_from_pgn(pgn, 1)
    X  = X + [np.zeros((8,8))] * (150-len(X))
    X=np.array(X).reshape(-1, 150, 8, 8)

    model= utils.load_model_gcp()

    white=int(model.predict(X)[0][0])
    black=int(model.predict(X)[0][1])

    return {"white":white,"black":black}
