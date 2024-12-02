from dl_logic import dl_models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from Utils import utils

bucket_name="chess_elo_prediction_lw1812"
pkl_df = utils.read_pickle_from_gcloud_df(bucket_name, gcloud_path="preprocessed_pgn/concat_result.pkl")
full_df = utils.read_parquet_from_gcloud_df(bucket_name, gcloud_path="pgn_time_increment_rating_data/cleaned_full_blitz_50000.parquet")

X = pkl_df.copy()
X = X.pgn

X_pad = pad_sequences(X, padding='post',maxlen=150, dtype= "int8")
y = full_df.loc[0:406894].white_rating

# Initializing model
model = dl_models.initialize_CNN_2D_model(np.array(X_pad[0]).shape)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_pad,y)

dl_models.train_model(model, X_train, y_train, ckp_filename='CNN_on_concat_pkl', epochs=100, validation_data=(X_test, y_test), patience=100)