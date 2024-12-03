from Utils import utils
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import pandas as pd
# import os

# from dl_logic import dl_models
# from Utils import matrix_creation
# import time
# from tqdm import tqdm
# tqdm.pandas()

if __name__ == "__main__":

    bucket_name="chess_elo_prediction_lw1812"
    # gcloud_filepath="pgn_time_increment_rating_data/full_evaluated_blitz_50000.parquet"
    # # gcloud_filepath = 'full_data/evaluated_blitz_50.parquet'

    # utils.upload_pickle_to_gcp(bucket_name, "pkl/X_matriced.pickle", "")
    source_folder = "preprocessed_pgn"
    destination_bucket_name = "mon-bucket-destination"
    destination_file_path = "preprocessed_pgn/concat_result.pkl"

    utils.concat_pkl_files_from_bucket(bucket_name, source_folder, bucket_name, destination_file_path)

    print('\nOpération terminée')