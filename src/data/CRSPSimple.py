import os
import numpy as np
import torch
import pandas as pd
import glob
from tqdm import tqdm

from data.metadata import crsp_stocks

class CRSPSimple(object):
    """

    
    """
    
    def __init__(self,
                 use_small_data: bool = False,
                 use_sample_data: bool = True,
                 fields: list=["close"],
                 all_years: bool = False,
                 tickers: list = crsp_stocks,
                 years: list=["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]):
        super().__init__()

        self.inputs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "inputs")
        self.use_sample_data = use_sample_data
        self.use_small_data = use_small_data
        self.fields = fields
        self.all_years = all_years
        self.years = years
        self.tickers = tickers

        self._read_data(years=self.years, fields=self.fields)


    def _read_data(self,
                   fields: list,
                   years: list):
        
        if self.use_small_data:
            crsp_df = pd.read_csv(os.path.join(self.inputs_path, "crsp_small_sample.csv"))

            crsp_df["date"] = pd.to_datetime(crsp_df["date"])
            crsp_df.set_index("date", inplace=True)            

        elif self.use_sample_data:
    
            crsp_df = pd.read_csv(os.path.join(self.inputs_path, "crsp_sample.csv"))

            crsp_df["date"] = pd.to_datetime(crsp_df["date"])
            crsp_df.set_index("date", inplace=True)
        else:
            if self.all_years:
                years = os.listdir(os.path.join(self.inputs_path, "US_CRSP_NYSE"))
                years = [val for val in years if val != ".DS_Store"]
                years.sort()

            crsp = []
            for y in tqdm(years, total=len(years), desc="Loading All CRSP Data"):
                files = glob.glob(os.path.join(self.inputs_path , "US_CRSP_NYSE", y, "*.csv.gz"))

                for f in files:
                    tmp_df = pd.read_csv(f,
                                        compression='gzip',
                                        on_bad_lines='skip')
                    tmp_df = tmp_df[["ticker"] + fields]
                    tmp_df["date"] = pd.to_datetime(f.split(os.sep)[-1].split(".")[0])

                    pivot_tmp_df = tmp_df.pivot_table(index=["date"], columns=["ticker"], values=fields)
                    pivot_tmp_df.index.name = None
                    pivot_tmp_df.columns = pivot_tmp_df.columns.droplevel(0)

                    crsp.append(pivot_tmp_df)
            crsp_df = pd.concat(crsp, axis=0)
            
            # dataset processing 1
            ## sort index
            crsp_df = crsp_df.sort_index()

            ## drop duplicates
            crsp_df = crsp_df.loc[~crsp_df.index.duplicated(keep='first')]

            ## resample (business days) and fill with zero
            crsp_df = crsp_df.resample("B").last().fillna(0)
            
            crsp_df.index.name = "date"

            # subset data
            if self.tickers is not None:
                crsp_df = crsp_df[self.tickers]

            # check if file exists
            if not os.path.exists(os.path.join(self.inputs_path, "crsp_simple_sample.csv")):
                crsp_df.to_csv(os.path.join(self.inputs_path, "crsp_simple_sample.csv"))

        # dataset processing 2
        ## compute returns and subset data
        returns_df = crsp_df.dropna().copy()

        # save indexes
        self.index = list(returns_df.index)
        self.columns = list(returns_df.columns)

        # create tensor with (num_nodes, num_features_per_node, num_timesteps)
        num_nodes = returns_df.shape[1]
        num_features_per_node = len(fields)
        num_timesteps = returns_df.shape[0]

        features = torch.zeros(num_nodes, num_features_per_node, num_timesteps)
        returns = torch.zeros(num_nodes, num_timesteps)
        for i in range(num_nodes):
            # features
            features[i, :, :] = torch.from_numpy(returns_df.loc[:, returns_df.columns[i]].values)

            # returns
            returns[i, :] = torch.from_numpy(returns_df.loc[:, returns_df.columns[i]].values)
        
        # create fully connected adjaneccny matrix
        A = torch.ones(num_nodes, num_nodes)

        self.A = A
        self.features = features
        self.returns = returns
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        loader = CRSPSimple(use_sample_data=False,
                            fields=["pvCLCL"],
                            all_years=False,
                            tickers=crsp_stocks,
                            years=["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"])
