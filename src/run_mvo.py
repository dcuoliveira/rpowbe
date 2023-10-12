import os
import pandas as pd
import torch
import argparse
import json
from tqdm import tqdm

from models.MVO import MVO
from data.CRSPSimple import CRSPSimple
from utils.dataset_utils import create_rolling_window_ts, check_bool
from loss_functions.SharpeLoss import SharpeLoss
from utils.conn_data import save_result_in_blocks

parser = argparse.ArgumentParser()

parser.add_argument('-mn', '--model_name', type=str, help='model name to be used for saving the model', default="mvo")
parser.add_argument('-nti', '--num_timesteps_in', type=int, help='size of the lookback window for the time series data', default=252 * 3)
parser.add_argument('-nto', '--num_timesteps_out', type=int, help='size of the lookforward window to be predicted', default=1)
parser.add_argument('--use_small_data', type=str, help='use small sample stocks data', default="False")
parser.add_argument('-usd', '--use_sample_data', type=str, help='use sample stocks data', default="True")
parser.add_argument('-ay', '--all_years', type=str, help='use all years to build dataset', default="False")
parser.add_argument('-lo', '--long_only', type=str, help='consider long only constraint on the optimization', default="False")
parser.add_argument('-meane', '--mean_estimator', type=str, help='name of the estimator to be used for the expected returns', default="mle")
parser.add_argument('-cove', '--covariance_estimator', type=str, help='name of the estimator to be used for the covariance of the returns', default="mle")

if __name__ == "__main__":

    args = parser.parse_args()

    model_name = args.model_name
    train_ratio = 0.6
    num_timesteps_in = args.num_timesteps_in
    num_timesteps_out = args.num_timesteps_out
    fix_start = False
    drop_last = True
    use_small_data = check_bool(args.use_small_data)
    use_sample_data = check_bool(args.use_sample_data)
    all_years = check_bool(args.all_years)
    long_only = check_bool(args.long_only)
    mean_estimator = args.mean_estimator
    covariance_estimator = args.covariance_estimator
    
    print("Running script with the following parameters: model_name: {}, use_small_data {}, use_sample_data: {}, all_years: {}, long_only: {}, mean_estimator: {}, covariance_estimator: {}".format(model_name, use_small_data, use_sample_data, all_years, long_only, mean_estimator, covariance_estimator))

    # add tag for long only or long-short portfolios
    model_name = "{model_name}_lo".format(model_name=model_name) if long_only else "{model_name}_ls".format(model_name=model_name)

    model_name = "{}_small".format(model_name) if use_small_data else model_name

    # add tag for sample data
    model_name = "{}_sample".format(model_name) if use_sample_data else model_name

    # add mean estimator tag to name
    model_name = "{model_name}_{mean_estimator}".format(model_name=model_name, mean_estimator=mean_estimator)

    # add covariance estimator tag to name
    model_name = "{model_name}_{covariance_estimator}".format(model_name=model_name, covariance_estimator=covariance_estimator)
    
    args.model_name = model_name

    # relevant paths
    source_path = os.path.dirname(__file__)
    inputs_path = os.path.join(source_path, "data", "inputs")

    # prepare dataset
    loader = CRSPSimple(use_small_data=use_small_data, use_sample_data=use_sample_data, all_years=all_years)
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
    model = MVO(mean_estimator=mean_estimator,
                covariance_estimator=covariance_estimator)

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
        "means": torch.vstack(model.means),
        "covs": torch.stack(model.covs),
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