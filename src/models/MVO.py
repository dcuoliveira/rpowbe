import torch
import numpy as np
import cvxopt as opt
from cvxopt import solvers

from estimators.Estimators import Estimators

class MVO(Estimators):
    def __init__(self,
                 risk_aversion: float=1,
                 mean_estimator: str="mle",
                 covariance_estimator: str="mle") -> None:
        """"
        This function impements the mean-variance optimization (MVO) method proposed by Markowitz (1952).

        Args:
            risk_aversion (float): risk aversion parameter. Defaults to 0.5. 
                                   The risk aversion parameter is a scalar that controls the trade-off between risk and return.
                                   According to Ang (2014), the risk aversion parameter of a risk neutral individual ranges from 1 and 10.
            mean_estimator (str): mean estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.
            covariance_estimator (str): covariance estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.

        References:
        Markowitz, H. (1952) Portfolio Selection. The Journal of Finance.
        Ang, Andrew, (2014). Asset Management: A Systematic Approach to Factor Investing. Oxford University Press. 
        """
        super().__init__()
        
        self.risk_aversion = risk_aversion
        self.mean_estimator = mean_estimator
        self.covariance_estimator = covariance_estimator

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                verbose: bool=True) -> torch.Tensor:
        
        solvers.options['show_progress'] = not verbose

        n = returns.shape[1]

        # mean estimator
        if self.mean_estimator == "mle":
            mean_t = self.MLEMean(returns)
        else:
            raise NotImplementedError

        # covariance estimator
        if self.covariance_estimator == "mle":
            cov_t = self.MLECovariance(returns)
        else:
            raise NotImplementedError

        # constraint 1: w_i >= 0 <=> -w_i <= 0, for all i
        c1 = torch.eye(n) * -1
        h = torch.zeros((n, 1))

        # constraint 2: \sum w_i = 1
        c2 = torch.ones((1, n))
        b = 1.0

        # convert to cvxopt matrices
        P = opt.matrix(cov_t.numpy().astype(np.double))
        q = opt.matrix(mean_t.numpy().astype(np.double))
        G = opt.matrix(c1.numpy().astype(np.double))
        h = opt.matrix(h.numpy().astype(np.double)) 
        A = opt.matrix(c2.numpy().astype(np.double))
        b = opt.matrix(b)

        # solve the problem
        opt_output = solvers.qp(P=(self.risk_aversion * P), q=(-1 * q), G=G, h=h, A=A, b=b) # minimizes the objective
        wt = torch.tensor(np.array(opt_output["x"])).T.repeat(num_timesteps_out, 1)

        return wt
    
    def forward_analytic(self,
                         returns: torch.Tensor,
                         num_timesteps_out: int) -> torch.Tensor:
        pass