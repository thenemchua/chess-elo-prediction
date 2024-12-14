from Utils import utils
import os
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
    # output_dir = "cleaned_data"
    # # # gcloud_filepath = 'full_data/evaluated_blitz_50.parquet'
    
    # df = utils.read_parquet_from_gcloud_df(bucket_name, gcloud_filepath)
    # cleaned_df = utils.clean_illegal_games(df, "pgn")
    
    # # Création du répertoire de sortie s'il n'existe pas
    # os.makedirs(output_dir, exist_ok=True)
    # filepath = os.path.join('checkpoint', "CNN_on_concat_pkl.model.keras")
    # cleaned_df.to_parquet(filepath)
    # print("saved df")
    
    
    # utils.upload_parquet_to_gcp(bucket_name, filepath)
    # print("uploaded to gcp")
    
    # utils.upload_pickle_to_gcp(bucket_name, "pkl/X_matriced.pickle", "")
    source_folder = "new_preprocessed_pgn"
    # destination_bucket_name = "mon-bucket-destination"
    destination_file_path = "new_preprocessed_pgn/onemillion_concat_result.pkl"

    utils.concat_pkl_files_from_bucket(bucket_name, source_folder, bucket_name, destination_file_path)

    print('\nOpération terminée')