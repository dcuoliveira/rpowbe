import os
import pandas as pd
import torch
import argparse
import json

from models.EW import EW
from data.CRSPSimple import CRSPSimple
from utils.conn_data import save_result_in_blocks

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, help='model name to be used for saving the model', default="ew")
parser.add_argument('--use_sample_data', type=bool, help='use sample stocks data', default=True)
parser.add_argument('--all_years', type=bool, help='use all years to build dataset', default=False)

if __name__ == "__main__":

    args = parser.parse_args()

    print("Running script with the following parameters: model_name: {}, use_sample_data: {}, all_years: {}".format(args.model_name, args.use_sample_data, args.all_years))

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")
    model_name = "{}_lo".format(args.model_name)

    model_name = "{}_sample".format(model_name) if args.use_sample_data else model_name

    # prepare dataset
    loader = CRSPSimple(use_sample_data=args.use_sample_data, all_years=args.all_years)
    prices = loader.prices.T
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