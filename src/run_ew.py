import os
import pandas as pd
import argparse

from models.EW import EW
from data.ETFsLoader import ETFsLoader
from utils.conn_data import save_result_in_blocks

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, help='model name to be used for saving the model', default="ew")

if __name__ == "__main__":

    args = parser.parse_args()

    model_name = args.model_name

    print("Running script with the following parameters: model_name: {}".format(model_name))

    model_name = "{}_lo".format(model_name)
    args.model_name = model_name

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")

    # prepare dataset
    loader = ETFsLoader()
    returns = loader.returns.T
    features = loader.features
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2]).T    

    # call model
    model = EW()

    # compute weights
    weights = model.forward(returns)

    # save results
    returns_df = pd.DataFrame(returns.numpy(), index=loader.index, columns=loader.columns)
    weights_df = pd.DataFrame(weights.numpy(), index=loader.index, columns=loader.columns)
    
    melt_returns_df = returns_df.reset_index().melt("index").rename(columns={"index": "date", "variable": "ticker", "value": "returns"})
    melt_weights_df = weights_df.reset_index().melt("index").rename(columns={"index": "date", "variable": "ticker", "value": "weights"})
    summary_df = pd.merge(melt_returns_df, melt_weights_df, on=["date", "ticker"], how="inner")

    results = {
        
        "model": model,
        "means": None,
        "covs": None,
        "train_loss": None,
        "eval_loss": None,
        "test_loss": None,
        "returns": returns_df,
        "weights": weights_df,
        "summary": summary_df

        }
    
    output_path = os.path.join(os.path.dirname(__file__),
                               "data",
                               "outputs",
                               model_name)
    
    save_result_in_blocks(results=results, args=args, path=output_path)