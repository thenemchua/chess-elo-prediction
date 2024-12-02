from fastapi import FastAPI
from model.model import forecast
from tensorflow import keras
from Utils import preprocessing
from Utils import matrix_creation
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


app = FastAPI()

# define root
# first endpoint
@app.get("/")
def status():
    return {"API": "connected"}

@app.get("/predict")
def predict(X):

    X=preprocessing.pgn_from_chess_com(X)
    df=pd.DataFrame({"pgn" : [X]})

    df=df.pgn.apply(lambda x: matrix_creation.create_matrice_from_pgn(x,12))

    X_pad = pad_sequences(df, padding='post',maxlen=150, dtype= "int64")

    model= keras.models.load_model("gs://chess_elo_prediction_lw1812/models/CNN_for_test_1288.keras")

    prediction=model.predict(X_pad)[0][0]

    return {'forecast': prediction}
