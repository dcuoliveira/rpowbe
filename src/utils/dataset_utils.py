import glob as glob
import os
import pandas as pd
import torch

def check_bool(str):
    if str.lower() == "false":
        return False
    elif str.lower() == "true": 
        return True
    else:
        raise Exception("Invalid boolean value: {}".format(str))

def aggregate_results(path):
    files = glob.glob(os.path.join(path, "*.csv"))

    all_summary = []
    for f in files:
        summary = pd.read_csv(f)

        all_summary.append(summary)
    
    all_summary_df = pd.concat(all_summary)
    all_summary_df.sort_values("date", inplace=True)

    return all_summary_df.reset_index(drop=True)

def timeseries_train_test_split_online(X, y, train_ratio):
    train_size = int(X.shape[0] * train_ratio)

    X_train = X[:train_size, :]
    y_train = y[:train_size, :]
    X_test = X[train_size:, :]
    y_test = y[(train_size-1):, :]

    return X_train, X_test, y_train, y_test

def timeseries_train_test_split(X, y, train_ratio):
    train_size = int(len(X) * train_ratio)
    
    X_train = X[:train_size, :]
    y_train = y[:train_size, :]
    X_test = X[train_size:, :]
    y_test = y[train_size:, :]

    return X_train, X_test, y_train, y_test

def create_rolling_indices(num_timesteps_in, num_timesteps_out, n_timesteps, fix_start):
    
    # generate rolling window indices
    indices = []
    for i in range(n_timesteps - num_timesteps_out):

        if fix_start:
            if i == 0:
                indices.append((0, (i + num_timesteps_in)))
            else:
                if indices[-1][1] == (n_timesteps - num_timesteps_out):
                    continue
                indices.append((0,  indices[-1][1] + num_timesteps_out))
        else:
            if i == 0:
                indices.append((i, (i + num_timesteps_in)))
            else:
                if indices[-1][1] == (n_timesteps - num_timesteps_out):
                    continue
                indices.append((indices[-1][0] + num_timesteps_out,  indices[-1][1] + num_timesteps_out))

    return indices

def create_online_rolling_window_ts(target, features, num_timesteps_in, num_timesteps_out, fix_start=False, drop_last=True):
    """"
    This function is used to create the rolling window time series to be used on DL ex-GNN.

    One important thing to note is that, since we are in the context of sharpe ratio optimization,
    and we are assuming positions are being taken at the close price of the same day, we have to 
    increase our target variable (prices) by one time step so as to compute the sharpe ratio properly.
    """
        
    if features.shape[0] != target.shape[0]:
        raise Exception("Features and target must have the same number of timesteps")

    n_timesteps = features.shape[0]
    indices = create_rolling_indices(num_timesteps_in=num_timesteps_in,
                                     num_timesteps_out=num_timesteps_out,
                                     n_timesteps=n_timesteps,
                                     fix_start=fix_start)
    
    # use rolling window indices to subset data
    window_features, window_target = [], []
    for i, j in indices:
        window_features.append(features[i:j, :])
        window_target.append(target[(i + 1):(j + num_timesteps_out), :])

    if drop_last:
        window_features = window_features[:-1]
        window_target = window_target[:-1]

    return torch.stack(window_features), torch.stack(window_target)

def create_rolling_window_ts(target, features, num_timesteps_in, num_timesteps_out, fix_start=False, drop_last=True):
    """"
    This function is used to create the rolling window time series to be used on DL ex-GNN.

    One important thing to note is that, since we are in the context of sharpe ratio optimization,
    and we are assuming positions are being taken at the close price of the same day, we have to 
    increase our target variable (prices) by one time step so as to compute the sharpe ratio properly.
    """
        
    if features.shape[0] != target.shape[0]:
        raise Exception("Features and target must have the same number of timesteps")

    n_timesteps = features.shape[0]
    indices = create_rolling_indices(num_timesteps_in=num_timesteps_in,
                                     num_timesteps_out=num_timesteps_out,
                                     n_timesteps=n_timesteps,
                                     fix_start=fix_start)
    
    # use rolling window indices to subset data
    window_features, window_target = [], []
    for i, j in indices:

        window_features.append(features[i:j, :]) # val \in [i, j)
        window_target.append(target[(j + num_timesteps_out):(j + num_timesteps_out + 1), :]) # val \in [j + num_timesteps_out, j + num_timesteps_out + 1)

    if drop_last:
        window_features = window_features[:-1]
        window_target = window_target[:-1]

    return torch.stack(window_features), torch.stack(window_target)

def create_rolling_window_ts_for_graphs(target, features, num_timesteps_in, num_timesteps_out, fix_start=False):
    """"
    This function is used to create the rolling window time series to be used on GNNs.

    One important thing to note is that, since we are in the context of sharpe ratio optimization,
    and we are assuming positions are being taken at the close price of the same day, we have to 
    increase our target variable (prices) by one time step so as to compute the sharpe ratio properly.
    """
    if features.shape[-1] != target.shape[-1]:
        raise Exception("Features and target must have the same number of timesteps")

    n_timesteps = features.shape[2]
    indices = create_rolling_indices(num_timesteps_in=num_timesteps_in,
                                     num_timesteps_out=num_timesteps_out,
                                     n_timesteps=n_timesteps,
                                     fix_start=fix_start)

    # use rolling window indices to subset data
    window_features, window_target = [], []
    for i, j in indices:
        window_features.append((features[:, :, i : i + num_timesteps_in]).numpy())
        window_target.append((target[:, (i + num_timesteps_in - 1):j]).numpy())

    return window_features, window_target

def concatenate_prices_returns(prices, returns):
    prices_names = list(prices.columns)
    returns_names = list(returns.columns)

    names = list(set(prices_names) & set(returns_names))
    
    all = []
    for name in names:
        all.append(prices[[name]].rename(columns={name: "{} price".format(name)}))
        all.append(returns[[name]].rename(columns={name: "{} ret".format(name)}))
    all_df = pd.concat(all, axis=1)

    return all_df, names

DEBUG = False

if __name__ == "__main__":

    if DEBUG:

        import os
        import numpy as np

        num_timesteps_in = 100
        num_timesteps_out = 1
        test_ratio = 0.2

        # relevant paths
        source_path = os.getcwd()
        inputs_path = os.path.join(source_path, "src", "data", "inputs")

        # prepare dataset
        prices = pd.read_excel(os.path.join(inputs_path, "etfs-zhang-zohren-roberts.xlsx"))
        prices.set_index("date", inplace=True)
        returns = np.log(prices).diff().dropna()
        prices = prices.loc[returns.index]
        features, names = concatenate_prices_returns(prices=prices, returns=returns)
        idx = features.index
        returns = returns[names].loc[idx].values.astype('float32')
        prices = prices[names].loc[idx].values.astype('float32')
        features = features.loc[idx].values.astype('float32')  

        # define train and test datasets
        X_train, X_test, prices_train, prices_test = timeseries_train_test_split(features, prices, test_ratio=test_ratio)
        X_train, X_val, prices_train, prices_val = timeseries_train_test_split(X_train, prices_train, test_ratio=test_ratio) 

        X_train, prices_train = create_rolling_window_ts(features=X_train, 
                                                         target=prices_train,
                                                         num_timesteps_in=num_timesteps_in,
                                                         num_timesteps_out=num_timesteps_out)        

        