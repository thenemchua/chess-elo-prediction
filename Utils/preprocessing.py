import pandas as pd

def extract_opening(DataFrame):
    DataFrame["opening"] = DataFrame["opening"].str.extract(r'openings/([^\.]+)')
    return DataFrame
