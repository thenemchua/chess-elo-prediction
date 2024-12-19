from dl_logic import dl_models_final
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from Utils import utils
import numpy as np
import os

# # GCP BUCKET
bucket_name="chess_elo_prediction_lw1812"

# Loading data from GCP
print('Loading data from GCP...')
pkl_df = utils.read_pickle_from_gcloud_df(bucket_name, gcloud_path="preprocessed_pgn/concat_result.pkl")
full_df = utils.read_parquet_from_gcloud_df(bucket_name, gcloud_path="pgn_time_increment_rating_data/cleaned_full_blitz_50000.parquet")
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


print('\nInitializing model...')
# Initializing model
double_cnn = dl_models_final.initialize_double_CNN(input_shape=(8, 8, 1), time_per_move_shape=(1,), learning_rate=.1)

print('\nModel initialized!')


print('\nTraining model...')
dl_models_final.train_model(double_cnn, [X_pad, time_pad], y, ckp_filename='alain_double_CNN_on_black', epochs=27, validation_split=.2, patience=100)
filepath = os.path.join('checkpoint', "alain_double_CNN_on_black.model.keras")
utils.upload_parquet_to_gcp(bucket_name, filepath, 'models/alain_double_CNN.model.keras')

print('\nEverything done!')
