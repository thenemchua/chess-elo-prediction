'''
This file contains all the deep learning models, including initialization and compiling used in the final product
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
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization,GRU,Attention
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Masking
from sklearn.model_selection import train_test_split
from Utils import preprocessing
import pandas as pd
from tensorflow.keras.regularizers import l2



# Modèle CNN 2D + LSTM permettant d'obtenir les meilleurs résultats

def init_cnn_lstm(input_shape, time_per_move_shape, learning_rate=0.01):
    # Inputs
    input_board = Input(shape=(None,) + input_shape)  # Variable number of games/sequences
    input_time = Input(shape=(None,) + time_per_move_shape)  # Time per move

    # Layer 1
    x = TimeDistributed(layers.Conv2D(128, (2,2), activation="leaky_relu"))(input_board)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = TimeDistributed(layers.AveragePooling2D(pool_size=(1,1)))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)

    # Layer 2
    x = TimeDistributed(layers.Conv2D(64, (2,2), activation="leaky_relu"))(x)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = TimeDistributed(layers.AveragePooling2D(pool_size=(1,1)))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)

    # Layer 3
    x = TimeDistributed(layers.Conv2D(32, (2,2), activation="leaky_relu"))(x)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = TimeDistributed(layers.AveragePooling2D(pool_size=(1,1)))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)

    # Layer 4
    x = TimeDistributed(layers.Conv2D(16, (1,1), activation="leaky_relu"))(x)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = TimeDistributed(layers.AveragePooling2D(pool_size=(1,1)))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)

    # Flatten CNN features
    x = TimeDistributed(layers.Flatten())(x)

    # Concatenate CNN features with time per move
    x = layers.Concatenate()([x, input_time])

    # Masking to ignore padding values
    x = Masking(mask_value=0)(x)

    # LSTM Layers
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    output = layers.Dense(1, activation='linear')(x)

    # Create model with multiple inputs
    model = models.Model(
        inputs=[input_board, input_time],
        outputs=output
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model


# Modèle CNN 2D sans LSTM permettant d'obtenir un résultat convenable en phase initiale de production

def initialize_double_CNN(input_shape, time_per_move_shape, learning_rate=0.01):
    """
    Modele double CNN >> Premier CNN puis on concatene le temps qui passe dans des layers de convolution.
    """
    # Inputs
    input_board = Input(shape=(None,) + input_shape)  # Variable number of games/sequences
    input_time = Input(shape=(None,) + time_per_move_shape)  # Time per move

    # Layer 1
    x = TimeDistributed(layers.Conv2D(128, (1,1), activation="leaky_relu"))(input_board)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = TimeDistributed(layers.AveragePooling2D(pool_size=(1,1)))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)

    # Layer 2
    x = TimeDistributed(layers.Conv2D(64, (1,1), activation="leaky_relu"))(x)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = TimeDistributed(layers.AveragePooling2D(pool_size=(1,1)))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)

    # Layer 3
    x = TimeDistributed(layers.Conv2D(32, (1,1), activation="leaky_relu"))(x)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = TimeDistributed(layers.AveragePooling2D(pool_size=(1,1)))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)

    # Layer 4
    x = TimeDistributed(layers.Conv2D(16, (1,1), activation="leaky_relu"))(x)
    x = TimeDistributed(layers.BatchNormalization())(x)
    x = TimeDistributed(layers.AveragePooling2D(pool_size=(1,1)))(x)
    x = TimeDistributed(layers.Dropout(0.2))(x)

    # Flatten CNN features
    x = TimeDistributed(layers.Flatten())(x)

    # # Aplatir input_time
    # time_flattened = layers.Flatten()(input_time)

    # Concatenate CNN features with time per move
    x = layers.Concatenate()([x, input_time])

    # CNN Layers
    x = layers.Conv1D(32, 1, activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(pool_size=1)(x)
    x = layers.Dropout(0.2)(x)

    # Global pooling temporel
    x = layers.GlobalAveragePooling1D()(x)

    # Output layer
    output = layers.Dense(1, activation='linear')(x)

    # Create model with multiple inputs
    model = models.Model(
        inputs=[input_board, input_time],
        outputs=output
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])

    return model


# définition du .fit afin d'intégrer l'early stopping et sauvegarde du modèle lors de l'entrainement

def train_model(model, X , y, ckp_filename, epochs=100, batch_size=32, patience=2, validation_data=None, validation_split=None):
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


# définition de l'évalutaiton afin d'accélérer les tests

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
