import os
import pandas as pd
import torch
import argparse
import json

from models.EW import EW
from data.NewETFs import NewETFs

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--model_name', type=str, help='model name to be used for saving the model', default="ew")

if __name__ == "__main__":

    args = parser.parse_args()

    model_name = args.model_name

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")

    # prepare dataset
    loader = NewETFs(use_last_data=True, use_first_50_etfs=True)
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

    # check if dir exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # save args
    args_dict = vars(args)  
    with open(os.path.join(output_path, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp)

    # save results
    output_name = "{model_name}.pt".format(model_name=model_name)
    torch.save(results, os.path.join(output_path, output_name))

    summary_df.to_csv(os.path.join(output_path, "summary.csv"), index=False)