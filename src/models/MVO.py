import torch
import numpy as np
import scipy.optimize as opt


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

    def objective(self,
                  weights):
        
        return -(np.dot(self.mean_t, weights) + ((self.risk_aversion) * np.dot(weights, np.dot(self.cov_t, weights))))

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True) -> torch.Tensor:
        
        K = returns.shape[1]

        # mean estimator
        if self.mean_estimator == "mle":
            self.mean_t = self.MLEMean(returns)
        elif (self.mean_estimator == "cbb") or (self.mean_estimator == "nobb"):
            self.mean_t = self.DependentBootstrapMean(returns=returns,
                                                 boot_method=self.mean_estimator,
                                                 Bsize=50,
                                                 rep=1000)
        elif self.mean_estimator == "rbb":
            self.mean_t = self.DependentBootstrapMean(returns=returns,
                                                 boot_method=self.mean_estimator,
                                                 Bsize=50,
                                                 rep=1000,
                                                 max_p=50,
                                                 max_q=50)
        else:
            raise NotImplementedError

        # covariance estimator
        if self.covariance_estimator == "mle":
            self.cov_t = self.MLECovariance(returns)
        elif (self.covariance_estimator == "cbb") or (self.covariance_estimator == "nobb"):
            self.cov_t = self.DependentBootstrapCovariance(returns=returns,
                                                      boot_method=self.covariance_estimator,
                                                      Bsize=50,
                                                      rep=1000)
        elif self.covariance_estimator == "rbb":
            self.cov_t = self.DepenBootstrapCovariance(returns=returns,
                                                  boot_method=self.covariance_estimator,
                                                  Bsize= 50,
                                                  rep=1000,
                                                  max_p= 50,
                                                  max_q= 50)
        else:
            raise NotImplementedError

        if long_only:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # the weights sum to one
            ]
            bounds = [(0, None) for _ in range(K)]
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # the weights sum to zero
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}  # the sum of absolute weights is one
            ]
            bounds = [(-1, 1) for _ in range(K)]

        # initial guess for the weights
        x0 = np.ones(K) / K

        # perform the optimization
        opt_output = opt.minimize(self.objective, x0, constraints=constraints, bounds=bounds, method='SLSQP')
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt
    
    def forward_analytic(self,
                         returns: torch.Tensor,
                         num_timesteps_out: int) -> torch.Tensor:
        pass