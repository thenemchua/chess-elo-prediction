from Utils import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
from tensorflow.keras.callbacks import EarlyStopping


from dl_logic import dl_models
from Utils import matrix_creation
import time
from tqdm import tqdm
tqdm.pandas()

if __name__ == "__main__":

    bucket_name="chess_elo_prediction_lw1812"
    gcloud_filepath="pgn_time_increment_rating_data/full_evaluated_blitz_50000.parquet"
    # gcloud_filepath = 'full_data/evaluated_blitz_50.parquet'

    print('Reading file from gcp...')

    df=utils.read_parquet_from_gcloud_df(bucket_name,gcloud_filepath)
    X = df[["pgn"]]
    # X = dl_models.create_X_from_initial_data_for_baseline(df)
    y=dl_models.create_y_from_initial_data_for_baseline(df)

    print('\nX,y initialized!')

    print('\nInitializing model...')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model= dl_models.initialize_baseline_model_3(X_train,embedding_dim=10,dropout_rate=0.25)
    print('\nModel initialized')
    
    print('\nTraining model...')
    model, history = dl_models.train_CNN_LSTM_model(model, X_train, y_train, 'baseline_model_50000', 10)

    print('\nOpération terminée')