'''
This file contains all the deep learning models, including initialization and compiling
'''
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.callbacks import EarlyStopping

import numpy as np

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
