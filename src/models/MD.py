import torch
import numpy as np
import scipy.optimize as opt

from estimators.Estimators import Estimators

class MD(Estimators):
    def __init__(self,
                 covariance_estimator: str="mle") -> None:
        """"
        This function impements the maximum diversification portfolio (MD) method proposed by Choueifaty and Coignard (2008).

        Args:
            covariance_estimator (str): covariance estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.

        References:
        Choieifaty, Y., and Y. Coignard, (2008). Toward Maximum Diversification. Journal of Portfolio Management.
        """
        super().__init__()
        
        self.covariance_estimator = covariance_estimator

    def objective(self, weights):
   
        portfolio_volatility = np.sqrt(weights.T @ self.cov_t @ weights)
        weighted_volatilities = weights.T @ self.vol_t
        diversification_ratio = - weighted_volatilities / portfolio_volatility
        return diversification_ratio

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int) -> torch.Tensor:

        # covariance estimator
        if self.covariance_estimator == "mle":
            cov_t = self.MLECovariance(returns)
        else:
            raise NotImplementedError
        
        N = returns.shape[1]

        self.cov_t = cov_t.numpy()
        self.vol_t = torch.sqrt(torch.diag(cov_t))[:, None].numpy()

        # constraint 1 : \sum w_i = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # constraint 2 : w_i \in [0, 1]
        bounds = [(0, 1) for _ in range(N)]

        # initial guess for the weights (equal distribution)
        w0 = np.repeat(1 / N, N)

        # perform the optimization
        opt_output = opt.minimize(self.objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt
