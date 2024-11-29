import os
import requests

import pandas as pd

import csv
import json

import Utils.api_requests as api_requests
from Utils import utils
from Utils import chess_evaluation
from iso3166 import countries

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

from Utils import utils
import pandas as pd
import os
import sys
import numpy as np

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
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from Utils import tensor_chess
from Utils import preprocessing


from dl_logic import dl_models
from Utils import matrix_creation

%load_ext autoreload
%autoreload 2

if __name__ == "__main__":

    df=utils.read_parquet_from_gcloud_df("chess_elo_prediction_lw1812","full_data/full_evaluated_daily_10000.parquet")
    X=dl_models.create_X_from_initial_data_for_baseline(df)
    y=dl_models.create_y_from_initial_data_for_baseline(df)

    X=X.apply(lambda x: matrix_creation.create_matrice_from_pgn(x,12))

    X_pad = pad_sequences(X, padding='post',maxlen=150, dtype= "int64")

    X_train,X_test, y_train,y_test = train_test_split(X_pad, y, test_size=0.2)

    model = dl_models.initialize_CNN_LSTM_model(X_pad[0].shape)

    dl_models.train_CNN_LSTM_model(model, X_train , y_train, epochs=5, batch_size=32, patience=2, validation_data=None, validation_split=0.3)

    print('Opération terminée')
