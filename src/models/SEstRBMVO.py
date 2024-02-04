import torch
import numpy as np
import scipy.optimize as opt

from estimators.Estimators import Estimators
from functionals.Functionals import Functionals

class SestRBMVO(Estimators, Functionals):
    def __init__(self,
                 risk_aversion: float=1,
                 mean_cov_estimator: str="sest",
                 num_boot: int = 200,
                 alpha: float = 0.95,
                 mean_functional: str=None,
                 cov_functional: str=None) -> None:
        """"
        This function impements the robust version of the mean-variance optimization (MVO) method proposed by Markowitz (1952).
        It intends to be robust in two senses:
            1. It uses a bootstrap procedure to estimate confidence intervals of the mean and covariance of returns.
            2. It uses the alpha-percentile mean and covariance as a plug-in estimator for the mean and covariance of returns.

        Optimization can be done using: (a) gradient descent or (b) using the scipy.optimize.minimize function.

        Args:
            risk_aversion (float): risk aversion parameter. Defaults to 0.5. 
                                   The risk aversion parameter is a scalar that controls the trade-off between risk and return.
                                   According to Ang (2014), the risk aversion parameter of a risk neutral individual ranges from 1 and 10.
            mean_estimator (str): mean estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.
            covariance_estimator (str): covariance estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.
            num_boot (int): number of bootstrap samples to be used. Defaults to 200.
            alpha (float): percentile. Defaults to 0.95.
            mean_functional (str): functional to be used to rank the mean of the returns. Defaults to "means".
            cov_functional (str): functional to be used to rank the covariance of the returns. Defaults to "eigenvalues".
            
        References:
        Markowitz, H. (1952) Portfolio Selection. The Journal of Finance.
        Ang, Andrew, (2014). Asset Management: A Systematic Approach to Factor Investing. Oxford University Press. 
        """

        Estimators.__init__(self)

        Functionals.__init__(self, alpha=alpha)
        
        self.risk_aversion = risk_aversion
        self.mean_cov_estimator = mean_cov_estimator 
        self.covariance_estimator = mean_cov_estimator
        self.num_boot = num_boot
        self.estimated_means = list()
        self.estimated_covs = list()
        self.mean_functional = mean_functional
        self.cov_functional = cov_functional

        if self.mean_functional == self.cov_functional:
            self.functional = self.mean_functional

    def objective(self,
                  weights: torch.Tensor,
                  maximize: bool=True) -> torch.Tensor:
        
        c = -1 if maximize else 1
        
        return (np.dot(weights, self.mean_t) - ((self.risk_aversion/2) * np.sqrt(np.dot(weights, np.dot(self.cov_t, weights))) )) * c

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True):
             
        self.K = returns.shape[1]

        # mean and cov estimates
        if self.mean_cov_estimator == "mle":
            self.list_mean_covs = [self.MLEMean(returns), self.MLECovariance(returns)]
            
            # compute the means and eigenvalues, and select the alpha-percentile of them
            self.mean_t = self.apply_functional(x=[self.list_mean_covs[0]], func=self.mean_functional)
            self.cov_t = self.apply_functional(x=[self.list_mean_covs[1]], func=self.cov_functional)

        elif self.mean_cov_estimator == "sest":
            self.list_mean_covs = self.SEstimator_Mean_Covariance(returns=returns,
                                                                  boot_method=self.mean_cov_estimator,
                                                                  Bsize=50,
                                                                  rep=self.num_boot)
            
            # compute the means and eigenvalues, and select the alpha-percentile of them
            self.mean_t = self.apply_functional(x=[val[0] for val in self.list_mean_covs], func=self.mean_functional)
            print(self.mean_t.shape)
            self.cov_t = self.apply_functional(x=[val[1] for val in self.list_mean_covs], func=self.cov_functional)
            print(self.cov_t.shape)

        else:
            self.list_mean_covs = self.DependentBootstrapMean_Covariance(returns=returns,
                                                                         boot_method=self.mean_cov_estimator,
                                                                         Bsize=50,
                                                                         rep=self.num_boot)
        
            # compute the means and eigenvalues, and select the alpha-percentile of them
            self.mean_t = self.apply_functional(x=[val[0] for val in self.list_mean_covs], func=self.mean_functional)
            self.cov_t = self.apply_functional(x=[val[1] for val in self.list_mean_covs], func=self.cov_functional)

        if long_only:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # the weights sum to one
            ]
            bounds = [(0, 1) for _ in range( self.K )]

            w0 = np.random.uniform(0, 1, size= self.K )
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # the weights sum to zero
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1},  # the weights sum to zero
            ]
            bounds = [(-1, 1) for _ in range( self.K )]

            w0 = np.random.uniform(-1, 1, size= self.K )

        # perform the optimization
        opt_output = opt.minimize(self.objective, w0, constraints=constraints, bounds=bounds)
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt
