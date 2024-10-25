import os
import torch
import pandas as pd
import numpy as np

class ETFsLoader(object):
    """

    
    """
    
    def __init__(self, tickers: list=None):
        super().__init__()

        self.inputs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "inputs")
        self.tickers = tickers
        self._read_data()

    def _read_data(self, resample_freq: str="B", sep: str=";", export_data: bool=True):
        
        etfs_df = pd.read_csv(os.path.join(self.inputs_path, "etfs.csv"), sep=sep)
        etfs_df["date"] = pd.to_datetime(etfs_df["date"])
        etfs_df.set_index("date", inplace=True)

        if self.tickers is not None:
            etfs_df = etfs_df[self.tickers]

        # dataset processing 1
        ## sort index
        etfs_df = etfs_df.sort_index()

        ## resample data to business days
        resampled_raw_data = etfs_df.resample(resample_freq).last()

        ## fill missing values forward
        filled_resampled_raw_data = resampled_raw_data.ffill()

        ## drop rows with na
        filled_resampled_raw_data = filled_resampled_raw_data.dropna()

        ## compute log-returns
        returns_df = np.log(filled_resampled_raw_data).diff().dropna()

        # export data
        if export_data:
            returns_df.to_csv(os.path.join(self.inputs_path, "etf_returns.csv"), sep=sep)

        # save indexes
        self.index = list(returns_df.index)
        self.columns = list(returns_df.columns)

        # create tensor with (num_nodes, num_features_per_node, num_timesteps)
        num_nodes = returns_df.shape[1]
        num_features_per_node = 1
        num_timesteps = returns_df.shape[0]

        features = torch.zeros(num_nodes, num_features_per_node, num_timesteps)
        returns = torch.zeros(num_nodes, num_timesteps)
        for i in range(num_nodes):
            # features and returns are the same
            features[i, :, :] = torch.from_numpy(returns_df.loc[:, returns_df.columns[i]].values)

            # returns and features are the same
            returns[i, :] = torch.from_numpy(returns_df.loc[:, returns_df.columns[i]].values)
        
        # create fully connected adjaneccny matrix
        A = torch.ones(num_nodes, num_nodes)

        self.A = A
        self.features = features
        self.returns = returns
