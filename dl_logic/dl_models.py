'''
This file contains all the deep learning models, including initialization and compiling
'''
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization,GRU,Attention
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from Utils import preprocessing
import pandas as pd
from tensorflow.keras.regularizers import l2



def chatgpt_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    cnn = layers.TimeDistributed(models.Sequential([
    layers.Conv3D(128, (2,2,2), input_shape=input_shape, activation="leaky_relu"),
    layers.BatchNormalization(),
    layers.AveragePooling3D(pool_size=(1,1,1))
    ]))(inputs)

    lstm_out = layers.LSTM(128, return_sequences=False)(cnn)

    dense_out = layers.Dense(10, activation='linear')(lstm_out)

    # Construire le modèle
    model = tf.keras.Model(inputs=inputs, outputs=dense_out)

    # Compiler le modèle
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def initialize_CNN_model(input_shape, learning_rate=0.1):
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
    model.add(layers.Dense(1, activation='linear'))

    # # LSTM
    # # model.add(layers.Flatten())
    # model.add(layers.LSTM(20, return_sequences=False))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(1, activation='linear'))

    # Model Compiling
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model



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
    model.add(layers.Dense(input_shape[0], activation='linear'))

    # # LSTM
    # # model.add(layers.Flatten())
    # model.add(layers.LSTM(20, return_sequences=False))

    # model.add(layers.Flatten())
    # model.add(layers.Dense(1, activation='linear'))

    # Model Compiling
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model


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

def train_model(model, X , y, ckp_filename, epochs=100, batch_size=32, patience=2, validation_data=None, validation_split=0.3):
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_filepath = f'checkpoint/{ckp_filename}.model.keras'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_mae',
        mode='min',
        save_best_only=True)

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es,model_checkpoint_callback],
        verbose=1
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
    max_features=max_features_baseline(pd.DataFrame(X)) #changement
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



def fit_baseline_model(model,X,y,batch_size=32, epochs=3, validation_split=0.2,callbacks=None):
    X,tk=preprocessing_baseline_francois(X)
    print(X)
    history = model.fit(X,y,batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks)
    return history, model, tk


def predict_baseline_model(model,X,tk):
    X,tk=preprocessing_baseline_francois(X,tk)
    y_pred=model.predict(X)
    return y_pred

def initialize_baseline_model_2(X,max_len=250,embedding_dim=10,dropout_rate=0.3):

    max_features=max_features_baseline(pd.DataFrame(X))

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

    max_features=max_features_baseline(pd.DataFrame(X))

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

def max_features_baseline_pgn(X):
    all_pgn=[]
    for pgn in X:
        for ele in pgn.split(" "):
            all_pgn.append(ele)
    return len(set(all_pgn))


def prepro_df_to_model_baseline_jules(df):

    df["pgn_all"]=df["pgn"]
    X = df["pgn_all"]
    y=create_y_from_initial_data_for_baseline(df)

    return X,y


def train_LSTM_model_jules(model, X_train , y_train, epochs=100, batch_size=32, patience=2, validation_data=None, validation_split=0.3, model_name="Test"):

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
