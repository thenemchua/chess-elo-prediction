from dl_logic import dl_models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from Utils import utils
import numpy as np
import os

# GCP BUCKET
bucket_name="chess-elo"

# Loading data from GCP
print('Loading data from GCP...')
pkl_df = utils.read_pickle_from_gcloud_df(bucket_name, gcloud_path="full-data/concat_result.pkl")
full_df = utils.read_parquet_from_gcloud_df(bucket_name, gcloud_path="full-data/cleaned_full_blitz_50000.parquet")
print('\nData loaded !')

time = full_df.loc[0:400000].time_per_move
time_pad = pad_sequences(time, padding='post', maxlen=150, dtype='float')

print('\nCreating X and y...')
X = pkl_df.copy()
X = X.loc[0:400000].pgn

X_pad = pad_sequences(X, padding='post',maxlen=150, dtype= "int8")
y = full_df.loc[0:400000].white_rating
print('\nX,y initialized!')

reshaped_time = time_pad.reshape(time_pad.shape[0], time_pad.shape[1], 1)

cnn_lstm = dl_models.init_cnn_lstm(input_shape=(8, 8, 1), time_per_move_shape=(1,))

print('\nTraining model...')
dl_models.train_model(cnn_lstm, [X_pad, time_pad], y, ckp_filename='new_cnn_lstm_on_concat_result_pkl', epochs=15, validation_split=.2, patience=10)


print('\nEverything done!')
