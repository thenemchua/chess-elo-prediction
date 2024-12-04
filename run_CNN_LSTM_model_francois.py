from dl_logic import dl_models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from Utils import utils
import numpy as np

# GCP BUCKET
bucket_name="chess-elo"

# Loading data from GCP
print('Loading data from GCP...')
pkl_df = utils.read_pickle_from_gcloud_df(bucket_name, gcloud_path="full-data/concat_result.pkl")
full_df = utils.read_parquet_from_gcloud_df(bucket_name, gcloud_path="full-data/cleaned_full_blitz_50000.parquet")
print('\nData loaded !')

time = full_df.loc[0:9682].time_per_move
time_pad = pad_sequences(time, padding='post', maxlen=150, dtype='float')

print('\nCreating X and y...')
X = pkl_df.copy()
X = X.pgn

X_pad = pad_sequences(X, padding='post',maxlen=150, dtype= "int8")
y = full_df.loc[0:9682].white_rating
print('\nX,y initialized!')

reshaped_time = time_pad.reshape(time_pad.shape[0], time_pad.shape[1], 1)

cnn_lstm = dl_models.init_cnn_lstm(input_shape=(8, 8, 1), time_per_move_shape=(1,))

cnn_lstm.fit(
    [X_pad, reshaped_time],
    y,
    epochs=10,
    validation_split=.2
)