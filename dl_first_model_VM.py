import pandas as pd
import numpy as np
from Utils import chess_evaluation
from Utils import api_requests
from Utils import utils
from Utils import utils
import os
import sys
import multiprocessing
import csv
import json
import ijson
from collections import defaultdict
import re
import numpy as np
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from dl_logic import dl_models


def initial_dl_model(df) :

    X=dl_models.create_X_from_initial_data_for_baseline(df)
    y=dl_models.create_y_from_initial_data_for_baseline(df)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model= dl_models.initialize_baseline_model_2(X_train,embedding_dim=10)
    history, model, tk=dl_models.fit_baseline_model(model,X_train,y_train,batch_size=32, epochs=500, validation_split=0.2)
    result = dl_models.predict_baseline_model(model, X_test,tk)
    return result 

# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.show()
