'''
This file contains all the deep learning models, including initialization and compiling
'''
from tensorflow.keras import layers
from tensorflow.keras import models
# from tensorflow.callbacks import EarlyStopping

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from Utils import preprocessing
import pandas as pd

def initialize_CNN_LSTM_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # CNN

    # Layer 1
    model.add(layers.conv3D(64, (2,2,2), padding='same', input_shape=input_shape, activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling3D())

    # Layer 2
    model.add(layers.conv3D(64, (2,2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling3D)

    # Layer 3
    model.add(layers.conv3D(64, (2,2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling3D())

    # Layer 4
    model.add(layers.conv3D(64, (2,2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling3D())

    # LSTM
    model.add(layers.Bidirectional())


    # Output
    model.add(layers.Dense(2, activation='linear'))

    # Model Compiling
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    return model


def train_CNN_LSTM_model(model, X , y, epochs=100, batch_size=32, patience=2, validation_data=None, validation_split=0.3):
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    return model, history

def evaluate_CNN_LSTM_model(model, X, y, batch_size=32):
    """
    Evaluate trained model performance on the dataset
    """

    print(f"Evaluating model on {len(X)} rows...")

    if model is None:
        print(f"No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"Model evaluated, MAE: {round(mae, 2)}")

    return metrics


# BASELINE FRANCOIS

def create_X_from_initial_data_for_baseline(df):
    """
    Crée X à partir de la donnée de base.
    X: le pgn global en format string, chaque coup séparé par un espace vide.

    """
    df=preprocessing.extract_moves_and_times_pgn(df)
    X = df[["pgn_all"]]
    X= X["pgn_all"].apply(lambda x: " ".join(x))
    return X

def create_y_from_initial_data_for_baseline(df):
    """
    Crée y à partir de la donnée de base.
    y: le rating de white.
    """
    y=df[["white_rating"]]
    return y

def tokeniser_pgn(x, max_features):
    """
    Permet de tokeniser x, x étant en entrée un df contenant les pgn en format str avec un espace vide entre chaque coup joué.
    """

    tk = Tokenizer(num_words=max_features, filters=".,",oov_token=-1)
    tk.fit_on_texts(x)
    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')
    X_token = tk.texts_to_sequences(x)
    return X_token, tk

def pad_sequence_X(x_token, maxlen):
    """
    Transforme tous les X pour avoir la même taille
    """
    X = sequence.pad_sequences(x_token, maxlen=maxlen)
    return X

def max_len_baseline(X):
    all_list_pgn=[]
    for pgn in X["pgn_all"]:
        all_list_pgn.append(len(pgn))
    return max(all_list_pgn)

def max_features_baseline(X):
    all_pgn=[]
    for pgn in X["pgn_all"]:
        for ele in pgn.split(" "):
            all_pgn.append(ele)
    return len(set(all_pgn))

def preprocessing_baseline_francois(X,tk=None):
    """
    X input = df incluant uniquement X["pgn_all"]
    """
    # max_len=max_len_baseline(pd.DataFrame(X))
    max_features=max_features_baseline(pd.DataFrame(X))
    print(f'max_features: {max_features}')
    if tk:
        X=tk.texts_to_sequences(X)
        print("Use tk already fitted")
    else:
        X,tk=tokeniser_pgn(X,max_features)
    X=pad_sequence_X(X, 250)
    return X,tk


def initialize_baseline_model(X,max_len=250,embedding_dim=10):

    max_features=max_features_baseline(pd.DataFrame(X))

    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=max_features,output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(64,return_sequences=True))
    model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(64))
    # rajouter dense16 relu
    model.add(Dense(1, activation='linear'))
    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model



def fit_baseline_model(model,X,y,batch_size=32, epochs=3, validation_split=0.2):
    X,tk=preprocessing_baseline_francois(X)
    print(X)
    history = model.fit(X,y,batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    return history, model, tk


def predict_baseline_model(model,X,tk):
    X,tk=preprocessing_baseline_francois(X,tk)
    y_pred=model.predict(X)
    return y_pred

def initialize_baseline_model_2(X,max_len=250,embedding_dim=10):

    max_features=max_features_baseline(pd.DataFrame(X))

    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=max_features,output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
