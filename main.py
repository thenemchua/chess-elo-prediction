import os
import requests

import pandas as pd

import csv
import json

import Utils.api_requests as api_requests
from Utils import utils
from Utils import chess_evaluation
from iso3166 import countries

data_folder = "data/games"
file_prefix = "titled_games_2024-10"
output_dir="data/evaluated_data"

chess_evaluation.evaluate_games_in_directory(data_folder, output_dir, depth=15, workers=12)

print('Opération terminée')