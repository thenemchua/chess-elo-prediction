from fastapi import FastAPI
from model.model import forecast
from tensorflow import keras
from Utils import preprocessing
from Utils import matrix_creation
from Utils import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


app = FastAPI()

# define root
# first endpoint
@app.get("/")
def status():
    return {"API": "connected, test francois"}

@app.get("/predict")
def predict(X):

    # X=preprocessing.extract_moves_chess(X)
    # df=pd.DataFrame({"pgn" : [X]})

    # df=df.pgn.apply(lambda x: matrix_creation.create_matrice_from_pgn(x,12))

    # X_pad = pad_sequences(df, padding='post',maxlen=150, dtype= "int64")

    model= utils.load_model_gcp()

    # prediction=model.predict(X_pad)[0][0]

    return {"test":model.summary()}
