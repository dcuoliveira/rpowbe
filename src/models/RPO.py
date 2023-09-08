import torch
import numpy as np
import scipy.optimize as opt
from scipy.stats import chi2 # for ceria and Stubs

from estimators.Estimators import Estimators
from utils.diagnostics import compute_summary_statistics

class RPO(Estimators):
    def __init__(self,
                 risk_aversion: float=1.0,
                 omega_estimator: str="mle",
                 mean_estimator: str="mle",
                 covariance_estimator: str="mle",
                 uncertainty_aversion_estimator: str="yin-etal-2022") -> None:
        """"
        This function impements the Robust Portofolio Optimization with quadratic uncertainty set.
        Implementation done using: Yin, Perchet and Soupé (2021).

        Args:
            risk_aversion (float): risk aversion parameter. Defaults to 0.5. 
                                   The risk aversion parameter is a scalar that controls the trade-off between risk and return.
                                   According to Ang (2014), the risk aversion parameter of a risk neutral individual ranges from 1 and 10.
            omega_estimator: Uncertainty set estimator to be used (always quadratic). Defaults to "mle", which defines the maximum likelihood estimator.
            uncertainty (float): uncertainty parameter. Defaults to 1.
                                 This parameter controls the size of the elipsoidal uncertainty set.
            mean_estimator (str): mean estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.
            covariance_estimator (str): covariance estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.

        References:
        Markowitz, H. (1952) Portfolio Selection. The Journal of Finance.
        Ang, Andrew, (2014). Asset Management: A Systematic Approach to Factor Investing. Oxford University Press. 
        C. Yin, R. Perchet & F. Soupé (2021) A practical guide to robust portfolio. Quantitative Finance.
        """
        super().__init__()
        
        self.risk_aversion = risk_aversion
        self.uncertainty_aversion_estimator = uncertainty_aversion_estimator
        self.omega_estimator = omega_estimator
        self.mean_estimator = mean_estimator
        self.covariance_estimator = covariance_estimator
        self.means = list()
        self.covs = list()

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True) -> torch.Tensor:

        K = returns.shape[1]
        T = returns.shape[0]

        # mean estimator
        if self.mean_estimator == "mle":
            mean_t = self.MLEMean(returns)
        elif (self.mean_estimator == "cbb") or (self.mean_estimator == "nobb"):
            mean_t = self.DependentBootstrapMean(returns=returns,
                                                 boot_method=self.mean_estimator,
                                                 Bsize=50,
                                                 rep=1000)
        elif self.mean_estimator == "rbb":
            mean_t = self.DependentBootstrapMean(returns=returns,
                                                 boot_method=self.mean_estimator,
                                                 Bsize=50,
                                                 rep=1000,
                                                 max_p=50,
                                                 max_q=50)
        else:
            raise NotImplementedError
        self.means.append(mean_t)

        # covariance estimator
        if self.covariance_estimator == "mle":
            cov_t = self.MLECovariance(returns)
        elif (self.covariance_estimator == "cbb") or (self.covariance_estimator == "nobb"):
            cov_t = self.DependentBootstrapCovariance(returns=returns,
                                                      boot_method=self.covariance_estimator,
                                                      Bsize=50,
                                                      rep=1000)
        elif self.covariance_estimator == "rbb":
            cov_t = self.DepenBootstrapCovariance(returns=returns,
                                                  boot_method=self.covariance_estimator,
                                                  Bsize= 50,
                                                  rep=1000,
                                                  max_p= 50,
                                                  max_q= 50)
        else:
            raise NotImplementedError
        self.covs.append(cov_t)
        
        # uncertainty set estimator
        if self.omega_estimator == "mle":
            omega_t = self.MLEUncertainty(T,cov_t)
        else:
            raise NotImplementedError
        
        # uncertainty (\kappa) aversion estimator
        if self.uncertainty_aversion_estimator == "yin-etal-2022":
            sharpe_ratios = compute_summary_statistics(returns)
            uncertainty_aversion = sharpe_ratios["Sharpe"] / 2 
        elif self.uncertainty_aversion_estimator == "ceria-stubbs-2006":
            # Cummulative chi-square function  
            eta = 0.95 # Ceria-Stubbs do not comment on the value they used, hence I am setting 95%
            uncertainty_aversion = chi2.ppf(1 - eta, df = K)
        else:
            raise NotImplementedError

        # max w^{\top}\bar{u} - (\kappa)*\sqrt(w^{\top} \Omega w) - \frac{\lambda}{2}w^{\top} \Sigma w
        # Problem
        def objective(weights):
            return np.dot(mean_t, weights) - uncertainty_aversion*np.sqrt(np.dot(weights,np.dot(omega_t,weights))) - (self.risk_aversion/2)*np.dot(weights,np.dot(cov_t,weights))

        if long_only:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # The weights sum to one
            ]
            bounds = [(0, None) for _ in range(K)]
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # the weights sum to zero
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1},  # the sum of absolute weights is one
                {'type': 'eq', 'fun': lambda x: np.linalg.norm(x) - 1}  # the norm L2 is one
            ]
            bounds = None

        # initial guess for the weights
        x0 = np.ones(K) / K

        # Perform the optimization
        opt_output = opt.minimize(objective, x0, constraints=constraints, bounds=bounds, method='SLSQP')
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt
    
    def forward_analytic(self,
                         returns: torch.Tensor,
                         num_timesteps_out: int) -> torch.Tensor:
        pass