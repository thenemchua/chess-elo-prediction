from Utils import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dl_logic import dl_models
from Utils import matrix_creation


if __name__ == "__main__":

    bucket_name="chess_elo_prediction_lw1812"
    gcloud_filepath="pgn_time_increment_rating_data/full_evaluated_blitz_50000.parquet"

    df=utils.read_parquet_from_gcloud_df(bucket_name,gcloud_filepath)
    X = df["pgn"]
    y=dl_models.create_y_from_initial_data_for_baseline(df)

    X=X.apply(lambda x: matrix_creation.create_matrice_from_pgn(x,12))

    X_pad = pad_sequences(X, padding='post',maxlen=150, dtype= "int64")

    X_train,X_test, y_train,y_test = train_test_split(X_pad, y, test_size=0.2)

    model = dl_models.initialize_CNN_LSTM_model(X_pad[0].shape)

    dl_models.train_CNN_LSTM_model(model, X_train , y_train, epochs=5, batch_size=32, patience=2, validation_data=None, validation_split=0.3)

    print('Opération terminée')
