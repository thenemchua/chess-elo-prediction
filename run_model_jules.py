from Utils import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os

from dl_logic import dl_models
from Utils import matrix_creation
import time


if __name__ == "__main__":

    bucket_name="chess_elo_prediction_lw1812"
    gcloud_filepath="pgn_time_increment_rating_data/cleaned_full_blitz_50000.parquet"
    # gcloud_filepath = 'full_data/evaluated_blitz_50.parquet'

    print('Reading file from gcp...')

    df = utils.read_parquet_from_gcloud_df(bucket_name,gcloud_filepath)
    X,y = dl_models.prepro_df_to_model_baseline_jules(df)

    print('\nX,y initialized!')

    print('\nInitializing model...')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model= dl_models.initialize_baseline_model_3(X_train,embedding_dim=10,dropout_rate=0.25)
    print('\nModel initialized')

    print('\nTraining model...')
    model, history = dl_models.train_CNN_LSTM_model_jules(model, X_train , y_train, epochs=6, batch_size=32, patience=, validation_data=None, validation_split=0.2)

    print('\nOpération terminée')
