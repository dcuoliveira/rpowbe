import torch
import torch.nn as nn
import numpy as np

class PositiveRetRatio(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prices, weights, ascent=True):
        
        # asset returns
        asset_returns = torch.diff(torch.log(prices), dim=1)

        # portfolio returns
        portfolio_returns = torch.mul(weights, asset_returns)

        # portfolio ratio of days with positive return
        positive_ret_ratio = (portfolio_returns > 0).sum() / (portfolio_returns.shape[0] * portfolio_returns.shape[1] * portfolio_returns.shape[2])

        return positive_ret_ratio * (-1 if ascent else 1)
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        import pandas as pd
        import numpy as np
        from tqdm import tqdm
        import os
        import sys
        import torch.utils.data as data

        sys.path.append(os.path.join(os.getcwd(), "src"))

        from data.NewETFs import NewETFs
        from utils.dataset_utils import create_rolling_window_ts, timeseries_train_test_split
        from models.DLPO import DLPO

        # parameters
        num_timesteps_in = 50
        num_timesteps_out = 1
        train_ratio = 0.6
        fix_start = False
        train_shuffle = True
        batch_size = 10
        drop_last = True

        # relevant paths
        source_path = os.getcwd()
        inputs_path = os.path.join(source_path, "src", "data", "inputs")

        # prepare dataset
        loader = NewETFs(use_last_data=True)
        prices = loader.y.T
        features = loader.X
        features = features.reshape(features.shape[0], features.shape[1] * features.shape[2]).T

        # define train and test datasets
        X_train, X_val, prices_train, prices_val = timeseries_train_test_split(features, prices, train_ratio=train_ratio)
        X_val, X_test, prices_val, prices_test = timeseries_train_test_split(X_val, prices_val, train_ratio=0.5) 

        X_train, prices_train = create_rolling_window_ts(features=X_train, 
                                                        target=prices_train,
                                                        num_timesteps_in=num_timesteps_in,
                                                        num_timesteps_out=num_timesteps_out,
                                                        fix_start=fix_start)

        # define data loaders
        train_loader = data.DataLoader(data.TensorDataset(X_train, prices_train), shuffle=train_shuffle, batch_size=batch_size, drop_last=drop_last)

        # (1) model
        model = DLPO(input_size=1426,
                     output_size=1426,
                     hidden_size=64,
                     num_layers=1,
                     num_timesteps_out=num_timesteps_out,
                     batch_first=True)

        # (2) loss fucntion
        lossfn = PositiveRetRatio()
        
        (X_batch, prices_batch) = next(iter(train_loader))
                    
        # compute forward propagation
        weights_pred = model.forward(X_batch)

        # compute loss
        loss = lossfn(prices_batch, weights_pred, ascent=True)