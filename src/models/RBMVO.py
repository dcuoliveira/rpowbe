import torch
import numpy as np
import scipy.optimize as opt


from estimators.Estimators import Estimators

# Robust bootstrap MVO
class RBMVO(Estimators):
    def __init__(self,
                 risk_aversion: float=1,
                 mean_estimator: str="mle",
                 covariance_estimator: str="mle",
                 num_boot: int = 200) -> None:
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
        self.num_boot = num_boot
        self.list_mean_covs = list()

    def objective(self,
                  weights: torch.Tensor,
                  maximize: bool=True) -> torch.Tensor:
        
        c = -1 if maximize else 1
        
        return (np.dot(self.mean_t, weights) - ((self.risk_aversion) * np.dot(weights, np.dot(self.cov_t, weights)))) * c

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True) -> torch.Tensor:
        
        K = returns.shape[1]

        # mean estimator
        if self.mean_estimator == "mle":
            self.list_mean_covs = [self.MLEMean(returns)]
        else:
            self.list_mean_covs = self.DependentBootstrapMean_Covariance(returns=returns,
                                                        boot_method=self.mean_estimator,
                                                        Bsize=50,
                                                        rep=self.num_boot)
           

        theta = torch.Tensor(np.random.uniform(-1, 1, size = K))
        # Nick's optimization proposal
        step = 0.01 # -> for gradient descent
        alpha = 0.95
        eps = 1e-6
        num_iter = 100
        while num_iter != 0:

            # compute utilities for all bootstraps
            bootstrap_utilities = list()
            for idx in range(len(self.list_mean_covs)):
                mean_i,cov_i = self.list_mean_covs[idx]
                utility = torch.matmul(theta,mean_i) - self.risk_aversion*torch.matmul(theta,torch.matmul(cov_i,theta))
                bootstrap_utilities.append(utility.item())

            # sort the utilities  with index
            idxs_bootstrap_utilities_sorted = sorted(range(len(bootstrap_utilities)), key=lambda k: bootstrap_utilities[k])
            idx_IC = idxs_bootstrap_utilities_sorted[int(alpha*self.num_boot)]

            # compute the derivative
            mean_IC, cov_IC = self.list_mean_covs[idx_IC] 
            d_utility_theta = mean_IC - 2*self.risk_aversion*torch.matmul(cov_IC,theta)
            new_theta = theta + step*d_utility_theta
            if torch.sum(torch.abs(theta - new_theta)) < eps:
                print("convergence attained")
                break
            theta = new_theta
            num_iter = num_iter - 1

        return theta
    
    def forward_analytic(self,
                         returns: torch.Tensor,
                         num_timesteps_out: int) -> torch.Tensor:
        pass