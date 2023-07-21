import os
import numpy as np
import torch
import pandas as pd

from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from utils.dataset_utils import create_rolling_window_ts_for_graphs

class ETFsZZR(object):
    """
    This class implements the dataset used in Zhang, Zohren, and Roberts (2021)
    https://arxiv.org/abs/2005.13665 in to torch geomatric data loader format.
    The data consists of daily observatios of four etf prices and returns 
    concatenated together, from January 2000 to February 2023.
    
    """
    
    def __init__(self):
        super().__init__()
        self._read_data()

    def _read_data(self):
        prices = pd.read_excel(os.path.join(os.path.dirname(__file__), "inputs", "etfs-zhang-zohren-roberts.xlsx"))

        # prepare dataset
        prices.set_index("date", inplace=True)

        # compute returns and subset data
        returns = np.log(prices).diff().dropna()

        # sanity check
        idx = returns.dropna().index
        returns = returns.loc[idx]
        prices = prices.loc[idx]

        # create tensor with (num_nodes, num_features_per_node, num_timesteps)
        num_nodes = prices.shape[1]
        num_features_per_node = 2
        num_timesteps = prices.shape[0]

        X = torch.zeros(num_nodes, num_features_per_node, num_timesteps)
        y = torch.zeros(num_nodes, num_timesteps)
        for i in range(num_nodes):
            # features
            X[i, 0, :] = torch.from_numpy(prices.loc[:, prices.columns[i]].values)
            X[i, 1, :] = torch.from_numpy(returns.loc[:, returns.columns[i]].values)

            # target
            y[i, :] = torch.from_numpy(prices.loc[:, prices.columns[i]].values)
        
        # create fully connected adjaneccny matrix
        A = torch.ones(num_nodes, num_nodes)

        self.A = A
        self.X = X
        self.y = y

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self,  num_timesteps_in: int = 12, num_timesteps_out: int = 12):

        features, target = create_rolling_window_ts_for_graphs(target=self.y,
                                                               features=self.X,
                                                               num_timesteps_in=num_timesteps_in,
                                                               num_timesteps_out=num_timesteps_out)

        self.features = features
        self.targets = target

    def get_dataset(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12) -> StaticGraphTemporalSignal:
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(self.edges,
                                            self.edge_weights,
                                            self.features,
                                            self.targets)
        
        return dataset
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        loader = ETFsZZR()
        dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
