import pandas as pd
import os

crsp_stocks = list(pd.read_csv(os.path.join(os.path.dirname(__file__), "inputs", "crsp_example.csv"), usecols=["ticker"])["ticker"].unique())
