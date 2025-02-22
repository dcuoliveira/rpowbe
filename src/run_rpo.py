import os
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from copy import copy

from models.RPO import RPO
from data.ETFsLoader import ETFsLoader
from utils.dataset_utils import create_rolling_window_ts, check_bool
from loss_functions.SharpeLoss import SharpeLoss
from utils.conn_data import save_result_in_blocks

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--model_name', type=str, help='model name to be used for saving the model', default="rpo")
parser.add_argument('-nti', '--num_timesteps_in', type=int, help='size of the lookback window for the time series data', default = 252 * 1)
parser.add_argument('-nto', '--num_timesteps_out', type=int, help='size of the lookforward window to be predicted', default=1)
parser.add_argument('-lo', '--long_only', type=str, help='use all years to build dataset', default="True")
parser.add_argument('-meane', '--mean_estimator', type=str, help='name of the estimator to be used for the expected returns', default="mle")
parser.add_argument('-cove', '--cov_estimator', type=str, help='name of the estimator to be used for the covariance of the returns', default="mle")
parser.add_argument('-uae', '--uncertainty_aversion_estimator', type=str, help='name of the uncertainty aversion estimator to be used', default="ceria-stubbs-2006")

if __name__ == "__main__":

    args = parser.parse_args()

    args.model = copy(args.model_name)
    
    model_name = args.model_name
    num_timesteps_in = args.num_timesteps_in
    num_timesteps_out = args.num_timesteps_out
    fix_start = False
    drop_last = True
    long_only = check_bool(args.long_only)
    mean_estimator = args.mean_estimator
    covariance_estimator = args.cov_estimator
    uncertainty_aversion_estimator = args.uncertainty_aversion_estimator

    print("Running script with the following parameters: model_name: {}, long_only: {}, mean_estimator: {}, covariance_estimator: {}".format(model_name, long_only, mean_estimator, covariance_estimator))

    # add tag for long only or long-short portfolios
    model_name = "{model_name}_lo".format(model_name=model_name) if long_only else "{model_name}_ls".format(model_name=model_name)

    # add mean estimator tag to name
    model_name = "{model_name}_{mean_estimator}".format(model_name=model_name, mean_estimator=mean_estimator)

    # add covariance estimator tag to name
    model_name = "{model_name}_{covariance_estimator}".format(model_name=model_name, covariance_estimator=covariance_estimator)

    # uncertainty aversion estimator tag
    if uncertainty_aversion_estimator == "yin-etal-2022":
        uae_tag = "y2022"
    elif uncertainty_aversion_estimator == "ceria-stubbs-2006":
        uae_tag = "cs2006"
    else:
        raise Exception("uncertainty_aversion_estimator not recognized")    
    
    # add uncertainty aversion tag to name
    model_name = "{model_name}_{uae_tag}".format(model_name=model_name, uae_tag=uae_tag)

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")

    # prepare dataset
    loader = ETFsLoader()
    returns = loader.returns.T
    features = loader.features
    features = features.reshape(features.shape[0], features.shape[1] * features.shape[2]).T  

    X_steps, returns_steps = create_rolling_window_ts(features=features, 
                                                      target=returns,
                                                      num_timesteps_in=num_timesteps_in,
                                                      num_timesteps_out=num_timesteps_out,
                                                      fix_start=fix_start,
                                                      drop_last=drop_last)

    # (1) call model
    model = RPO(omega_estimator="mle",
                mean_estimator=mean_estimator,
                covariance_estimator=covariance_estimator,
                uncertainty_aversion_estimator=uncertainty_aversion_estimator)

    # (2) loss fucntion
    lossfn = SharpeLoss()

    # (3) training/validation + oos testing
    test_weights = torch.zeros((X_steps.shape[0], num_timesteps_out, X_steps.shape[2]))
    test_returns = torch.zeros((X_steps.shape[0], num_timesteps_out, X_steps.shape[2]))
    test_loss = torch.zeros((X_steps.shape[0], num_timesteps_out))
    pbar = tqdm(range(X_steps.shape[0]), total=(X_steps.shape[0] + 1))
    for step in pbar:
        X_t = X_steps[step, :, :]
        returns_t1 = returns_steps[step, :, :]

        weights_t1 = model.forward(returns=X_t, num_timesteps_out=num_timesteps_out, long_only=long_only)
        test_weights[step, :, :] = weights_t1

        loss = lossfn(weights=weights_t1, returns=returns_t1)
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
        
        "means": None,
        "covs": None,
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
    
    save_result_in_blocks(results=results, args=args, path=output_path)