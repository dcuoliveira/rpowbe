import pickle
import numpy as np

def compute_realized_ewma_vol(returns, window=50):
    
    ewma_vol_df = (returns.ewm(window).std() * np.sqrt(252))

    return ewma_vol_df

def save_pickle(path: str,
                obj: dict):

    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path: str):
    file = open(path, 'rb')
    target_dict = pickle.load(file)

    return target_dict