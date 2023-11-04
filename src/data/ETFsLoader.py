import os
import torch
import pandas as pd

class ETFsLoader(object):
    """

    
    """
    
    def __init__(self, tickers: list=None):
        super().__init__()

        self.inputs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "inputs")
        self.tickers = tickers
        self._read_data()

    def _read_data(self):
        
        etfs_df = pd.read_csv(os.path.join(self.inputs_path, "etfs.csv"))
        etfs_df["date"] = pd.to_datetime(etfs_df["date"])
        etfs_df.set_index("date", inplace=True)

        if self.tickers is not None:
            etfs_df = etfs_df[self.tickers]

        # dataset processing 1
        ## sort index
        etfs_df = etfs_df.sort_index()

        # dataset processing 2
        ## compute returns and subset data
        returns_df = etfs_df.dropna().copy()

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
            # features
            features[i, :, :] = torch.from_numpy(returns_df.loc[:, returns_df.columns[i]].values)

            # returns
            returns[i, :] = torch.from_numpy(returns_df.loc[:, returns_df.columns[i]].values)
        
        # create fully connected adjaneccny matrix
        A = torch.ones(num_nodes, num_nodes)

        self.A = A
        self.features = features
        self.returns = returns
