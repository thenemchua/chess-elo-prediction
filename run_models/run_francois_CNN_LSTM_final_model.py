from dl_logic import dl_models_final
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from Utils import utils
import numpy as np
import os
from tensorflow import keras

# GCP BUCKET
bucket_name="chess-elo"

# Loading data from GCP
print('Loading data from GCP...')
pkl_df = utils.read_pickle_from_gcloud_df(bucket_name, gcloud_path="full-data/concat_result.pkl")
full_df = utils.read_parquet_from_gcloud_df(bucket_name, gcloud_path="full-data/cleaned_full_blitz_50000.parquet")
print('\nData loaded !')

print('\nCreating X and y...')
# Initalizing X (one feature: PGN as matrixes (n,8,8))
X = pkl_df.copy()
X = X.pgn

time = full_df.loc[0:406894].time_per_move

# Padding sequences to input correct shape in the model
time_pad = pad_sequences(time, padding='post', maxlen=150, dtype='float')
X_pad = pad_sequences(X, padding='post',maxlen=150, dtype= "int8")

y = full_df.loc[0:406894].black_rating
print('\nX,y initialized!')

reshaped_time = time_pad.reshape(time_pad.shape[0], time_pad.shape[1], 1)

# cnn_lstm = dl_models.init_cnn_lstm(input_shape=(8, 8, 1), time_per_move_shape=(1,))
cnn_lstm = keras.models.load_model("checkpoint/new_cnn_lstm_on_concat_result_pkl.model.keras")

print('\nTraining model...')
dl_models_final.train_model(cnn_lstm, [X_pad, time_pad], y, ckp_filename='new_cnn_lstm_on_concat_result_pkl_2', epochs=15, validation_split=.2, patience=10)
bucket_name="chess-elo"
filepath = os.path.join('checkpoint', "new_cnn_lstm_on_concat_result_pkl_2.model.keras")
utils.upload_parquet_to_gcp(bucket_name, filepath,"models/cnn_lstm.model_2.keras")

print('\nEverything done!')
