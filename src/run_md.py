import os
import pandas as pd
import torch
import argparse
import json
from tqdm import tqdm

from models.MD import MD
from data.NewETFs import NewETFs
from utils.dataset_utils import create_rolling_window_ts
from loss_functions.SharpeLoss import SharpeLoss

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--model_name', type=str, help='model name to be used for saving the model', default="md")
parser.add_argument('-nti', '--num_timesteps_in', type=int, help='size of the lookback window for the time series data', default=252)
parser.add_argument('-nto', '--num_timesteps_out', type=int, help='size of the lookforward window to be predicted', default=1)

if __name__ == "__main__":

    args = parser.parse_args()

    model_name = args.model_name
    train_ratio = 0.6
    num_timesteps_in = args.num_timesteps_in
    num_timesteps_out = args.num_timesteps_out
    fix_start = False
    drop_last = True

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")

    # prepare dataset
    loader = NewETFs(use_last_data=True, use_first_50_etfs=True)
    prices = loader.prices.T
    returns = loader.returns.T
    features = loader.features
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2]).T  

    X_steps, prices_steps = create_rolling_window_ts(features=features, 
                                                     target=prices,
                                                     num_timesteps_in=num_timesteps_in,
                                                     num_timesteps_out=num_timesteps_out,
                                                     fix_start=fix_start,
                                                     drop_last=drop_last)

    # (1) call model
    model = MD()

    # (2) loss fucntion
    lossfn = SharpeLoss()

    # (3) training/validation + oos testing
    test_weights = torch.zeros((X_steps.shape[0], num_timesteps_out, X_steps.shape[2]))
    test_returns = torch.zeros((X_steps.shape[0], num_timesteps_out, X_steps.shape[2]))
    test_loss = torch.zeros((X_steps.shape[0], num_timesteps_out))
    pbar = tqdm(range(X_steps.shape[0]), total=(X_steps.shape[0] + 1))
    for step in pbar:
        X_t = X_steps[step, :, :]
        prices_t1 = prices_steps[step, :, :]

        weights_t1 = model.forward(returns=X_t, num_timesteps_out=num_timesteps_out)
        test_weights[step, :, :] = weights_t1

        loss, returns_t1 = lossfn(weights=weights_t1, prices=prices_t1)
        test_returns[step, :, :] = returns_t1

        test_loss[step, :] = loss.item()

        pbar.set_description("Steps: %d, Test sharpe : %1.5f" % (step, loss.item()))
    print("Avg sharpe : %1.5f" % (torch.mean(test_loss).item()))

    if test_weights.dim() == 3:
        weights = test_weights.reshape(test_weights.shape[0] * test_weights.shape[1], test_weights.shape[2])
    else:
        weights = test_weights

    if test_returns.dim() == 3:
        returns = test_returns.reshape(test_weights.shape[0] * test_weights.shape[1], test_weights.shape[2])
    else:
        returns = test_returns

    # (4) save results
    returns_df = pd.DataFrame(returns.numpy(), index=loader.index[(num_timesteps_in + num_timesteps_out):], columns=loader.columns)
    weights_df = pd.DataFrame(weights.numpy(), index=loader.index[(num_timesteps_in + num_timesteps_out):], columns=loader.columns)
    
    melt_returns_df = returns_df.reset_index().melt("index").rename(columns={"index": "date", "variable": "ticker", "value": "returns"})
    melt_weights_df = weights_df.reset_index().melt("index").rename(columns={"index": "date", "variable": "ticker", "value": "weights"})
    summary_df = pd.merge(melt_returns_df, melt_weights_df, on=["date", "ticker"], how="inner")

    results = {
        
        "model": model,
        "train_loss": None,
        "eval_loss": None,
        "test_loss": test_loss,
        "returns": returns_df,
        "weights": weights_df,
        "summary": summary_df

        }

    output_path = os.path.join(os.path.dirname(__file__),
                                    "data",
                                    "outputs",
                                    model_name)

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