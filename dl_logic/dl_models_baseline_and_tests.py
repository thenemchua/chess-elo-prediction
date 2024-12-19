'''
This file contains all the deep learning models, including initialization and compiling used during our exploratory phase.
It is a repository of test and baseline definition
'''

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from Utils import preprocessing
import pandas as pd



# définition de baseline score via 3 modèles LSTM se basant avec en donnée d'entrée sur le PGN au format Str uniquement

def initialize_baseline_model(X,max_len=250,embedding_dim=10):

    max_features=preprocessing.max_features_baseline(pd.DataFrame(X))

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



def initialize_baseline_model_2(X,max_len=250,embedding_dim=10,dropout_rate=0.3):

    max_features=preprocessing.max_features_baseline(pd.DataFrame(X))

    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=max_features,output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(64))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model



def initialize_baseline_model_3(X,max_len=250,embedding_dim=10,dropout_rate=0.15):

    max_features=preprocessing.max_features_baseline(pd.DataFrame(X))

    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=max_features,output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

# LSTM le plus performant grâce au Dropout intégré

def initialize_LSTM_model(input_shape=(150,1), learning_rate=0.1):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # model.add(layers.LSTM(20, return_sequences=True))
    # model.add(layers.LSTM(20, return_sequences=False))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(64))
    model.add(Dropout(0.15))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='linear'))

    # Model Compiling
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model

# Fit du modèle qui contient le preprocessing sur X_train

def fit_baseline_model(model,X,y,batch_size=32, epochs=3, validation_split=0.2,callbacks=None):
    X,tk=preprocessing.preprocessing_baseline_francois(X)
    print(X)
    history = model.fit(X,y,batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks)
    return history, model, tk

# Fit du modèle qui contient le preprocessing sur X_test

def predict_baseline_model(model,X,tk):
    X,tk=preprocessing.preprocessing_baseline_francois(X,tk)
    y_pred=model.predict(X)
    return y_pred


def train_LSTM_model_baseline(model, X_train , y_train, epochs=100, batch_size=32, patience=2, validation_data=None, validation_split=0.3, model_name="Test"):

    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_filepath = f'checkpoint/epoch{epochs}_{model_name}.model.keras'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_mae',
        mode='min',
        save_best_only=True)

    history, model, tk=fit_baseline_model(model,X_train,y_train,batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[es,model_checkpoint_callback])

    return model, history, tk



# Tests sur la partie CNN, d'abord en conv 3D via un input de matrices (12,8,8)

def initialize_CNN_3D_model(input_shape, learning_rate=0.1):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # CNN

    # Layer 1
    model.add(layers.Conv3D(128, (2,2,2), input_shape=input_shape, activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling3D(pool_size=(1,1,1)))
    model.add(Dropout(0.15))

    # Layer 2
    model.add(layers.Conv3D(64, (2,2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling3D(pool_size=(1,1,1)))

    # Layer 3
    model.add(layers.Conv3D(32, (2,2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling3D(pool_size=(1,1,1)))

    # Layer 4
    model.add(layers.Conv3D(16, (2,2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling3D(pool_size=(1,1,1)))
    model.add(Dropout(0.15))

    # Output CNN
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='linear'))

    # Model Compiling
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model

# Tests sur la partie CNN,en conv 2D via un input de matrices (8,8)

def initialize_CNN_2D_model(input_shape, learning_rate=0.1):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # CNN

    # Layer 1
    model.add(layers.Conv2D(128, (2,2), input_shape=input_shape, activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(1,1)))
    model.add(Dropout(0.15))

    # Layer 2
    model.add(layers.Conv2D(64, (2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(1,1)))

    # Layer 3
    model.add(layers.Conv2D(32, (2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(1,1)))

    # Layer 4
    model.add(layers.Conv2D(16, (2,2), activation="leaky_relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D(pool_size=(1,1)))
    model.add(Dropout(0.15))

    # Output CNN
    model.add(layers.Flatten())

    model.add(layers.Dense(2, activation='linear'))

    # Model Compiling
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model
