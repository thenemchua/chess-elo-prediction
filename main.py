import os
import requests

import pandas as pd

import csv
import json

import api_requests
from Utils import utils
from iso3166 import countries

data_folder = "data"
file_prefix = "titled_games_2024-10"
output_dir = 'data/games'
for f in os.listdir(data_folder):
    if f.startswith(file_prefix) and f.endswith(".json"):
        utils.split_large_json_stream(os.path.join('data', f), output_dir)

print('Opération terminée')