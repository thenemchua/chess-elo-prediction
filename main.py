import os
import requests

import pandas as pd

import csv
import json

import Utils.api_requests as api_requests
from Utils import utils
from Utils import chess_evaluation
from iso3166 import countries

if __name__ == "__main__":
    sizes = [50000]

    game_modes = ['blitz']
    for s in sizes:
        for mode in game_modes:
            input_dir = f'partial_data/{s}'
            output_dir = f'partial_full_data/{s}'
            utils.pd_reconstitute_full_parquet(input_dir, output_dir, mode)
            # utils.pd_reconstitue_partial_parquet(input_dir, output_dir, mode)

    print('Opération terminée')