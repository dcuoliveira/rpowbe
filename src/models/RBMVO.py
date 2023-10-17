import torch
import numpy as np
import scipy.optimize as opt

from estimators.Estimators import Estimators
from functionals.Functionals import Functionals

class RBMVO(Estimators, Functionals):
    def __init__(self,
                 risk_aversion: float=1,
                 mean_cov_estimator: str="mle",
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
        self.means = list()
        self.covs = list()
        self.mean_functional = mean_functional
        self.cov_functional = cov_functional

        if self.mean_functional == self.cov_functional:
            self.functional = self.mean_functional

    def objective(self,
                  weights: torch.Tensor,
                  maximize: bool=True) -> torch.Tensor:
        
        c = -1 if maximize else 1
        
        return (np.dot(self.mean_t, weights) - ((self.risk_aversion) * np.dot(weights, np.dot(self.cov_t, weights)))) * c

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True):
             
        K = returns.shape[1]

        # mean and cov estimates
        if self.mean_cov_estimator == "mle":
            self.list_mean_covs = [self.MLEMean(returns)]
        else:
            self.list_mean_covs = self.DependentBootstrapMean_Covariance(returns=returns,
                                                                         boot_method=self.mean_cov_estimator,
                                                                         Bsize=50,
                                                                         rep=self.num_boot)
        
        # compute the means and eigenvalues, and select the alpha-percentile of them
        self.mean_t = self.apply_functional(x=[val[0] for val in self.list_mean_covs], func=self.mean_functional)
        self.cov_t = self.apply_functional(x=[val[1] for val in self.list_mean_covs], func=self.cov_functional)

        # self.mean.append(self.mean_t)
        # self.cov.append(self.cov_t)

        if long_only:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # the weights sum to one
            ]
            bounds = [(0, None) for _ in range(K)]

            w0 = np.random.uniform(0, 1, size=K)
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # the weights sum to zero
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1},  # the weights sum to zero
            ]
            bounds = [(-1, 1) for _ in range(K)]

            w0 = np.random.uniform(-1, 1, size=K)

        # perform the optimization
        opt_output = opt.minimize(self.objective, w0, constraints=constraints, bounds=bounds, method='SLSQP')
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt

    def forward_gradient(self,
                         returns: torch.Tensor,
                         num_timesteps_out: int,
                         long_only: bool=True) -> torch.Tensor:
        
        self.K = returns.shape[1]

        # mean and cov estimates
        if self.mean_cov_estimator == "mle":
            self.list_mean_covs = [self.MLEMean(returns)]
        else:
            self.list_mean_covs = self.DependentBootstrapMean_Covariance(returns=returns,
                                                                         boot_method=self.mean_cov_estimator,
                                                                         Bsize=50,
                                                                         rep=self.num_boot)
           

        wt = torch.Tensor(np.random.uniform(-1, 1, size = self.K))
        # Nick's optimization proposal
        step = 0.01 # -> for gradient descent
        eps = 1e-6
        num_iter = 100
        while num_iter != 0:
            
            # compute the utilities and select the alpha-percentile
            mean_IC, cov_IC = self.apply_functional(x=self.list_mean_covs, func=self.functional)

            # self.mean.append(self.mean_t)
            # self.cov.append(self.cov_t)

            # # compute the derivative
            d_utility_theta = mean_IC - 2*self.risk_aversion*torch.matmul(cov_IC, wt)
            new_wt = wt + step*d_utility_theta
            if torch.sum(torch.abs(wt - new_wt)) < eps:
                print("convergence attained")
                break
            wt = new_wt
            num_iter = num_iter - 1

        wt = wt.repeat(num_timesteps_out, 1)

        return wt