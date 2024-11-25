import os
import requests

import pandas as pd

import csv
import json

import Utils.api_requests as api_requests
from Utils import utils
from Utils import chess_evaluation
from iso3166 import countries

data_folder = "evaluated_data"
output_dir="evaluated_data"

chess_evaluation.evaluate_games_in_directory(data_folder, output_dir, depth=1, workers=1)

print('Opération terminée')