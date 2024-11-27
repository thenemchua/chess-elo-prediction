import os
import requests

import pandas as pd

import csv
import json

import Utils.api_requests as api_requests
from Utils import utils
from Utils import chess_evaluation
from iso3166 import countries



sizes = [10000, 50000]
bucket_name = "chess_elo_prediction_lw1812"
for size in sizes:
    utils.reconstitute_json_in_gcloud(bucket_name, size)

print('Opération terminée')