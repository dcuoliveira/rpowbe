import pandas as pd
import torch

from loss_functions.ExpectedRet import ExpectedRet
from loss_functions.Volatility import Volatility
from loss_functions.Sharpe import Sharpe
from loss_functions.Sortino import Sortino
from loss_functions.AverageDD import AverageDD
from loss_functions.MaxDD import MaxDD
from loss_functions.PositiveRetRatio import PositiveRetRatio

def compute_summary_statistics(portfolio_returns: torch.tensor,
                               default_metrics: list = [ExpectedRet, Volatility, Sharpe, Sortino, AverageDD, MaxDD, PositiveRetRatio]):
    
    portfolio_stats = {}
    for metric in default_metrics:
        metric = metric()
        portfolio_stats[metric.name] = metric.forward(returns=portfolio_returns).item()

    return portfolio_stats

DEBUG = False

if __name__ == "__main__":

    if DEBUG:
        import os
        import pandas as pd

        model = "ew"
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "data",
                                   "outputs",
                                   model)
        
        summary = pd.read_csv(os.path.join(output_path, "summary.csv"))
        summary["pnl"] = summary["returns"] * summary["weights"]
        summary["model"] = model

        cum_pnl_df = torch.tensor(summary.loc[summary["model"] == model].groupby("date").sum()["pnl"])

        stats = compute_summary_statistics(portfolio_returns=cum_pnl_df)