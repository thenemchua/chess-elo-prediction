from Utils import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os

from dl_logic import dl_models
from Utils import matrix_creation
import time


if __name__ == "__main__":

    # bucket_name="chess_elo_prediction_lw1812"
    # gcloud_filepath="evaluated_data/50000/evaluated_blitz_50000_part_1.json"
    # # gcloud_filepath = 'full_data/evaluated_blitz_50.parquet'

    print('Reading file from gcp...')

    df = pd.read_parquet("data/full_data_full_evaluated_daily_10000.parquet")
    y= dl_models.create_y_from_initial_data_for_baseline(df)
    X=dl_models.create_X_from_initial_data_for_baseline(df)

    print('\nX,y initialized!')

    print('\nInitializing model...')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model= dl_models.initialize_baseline_model_3(X_train,embedding_dim=10,dropout_rate=0.25)
    print('\nModel initialized')

    print('\nTraining model...')
    model, history, tk = dl_models.train_CNN_LSTM_model_jules(model, X_train , y_train, epochs=100, batch_size=32, patience=10, validation_data=None, validation_split=0.2,model_name="Test_10000_daily")

    print('\nOpération terminée')
