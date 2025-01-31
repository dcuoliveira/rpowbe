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
        This model was originally proposed by Cerias and Stubbs (2016) and later studied by Yin, Perchet and Soupé (2021).

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
        Ceria and Stubbs (2006) Robust Portofolio Optimization. Management Science.
        C. Yin, R. Perchet & F. Soupé (2021) A practical guide to robust portfolio. Quantitative Finance.
        """
        super().__init__()
        
        self.risk_aversion = risk_aversion
        self.uncertainty_aversion_estimator = uncertainty_aversion_estimator
        self.omega_estimator = omega_estimator
        self.mean_estimator = mean_estimator
        self.covariance_estimator = covariance_estimator
        self.estimated_means = list()
        self.estimated_covs = list()

    def objective(self,
                  weights: torch.Tensor,
                  maximize: bool=True) -> torch.Tensor:
        
        c = -1 if maximize else 1
        
        # Problem       
        # max w^{\top}\bar{u} - (\kappa)*\sqrt(w^{\top} \Omega w) - \frac{\lambda}{2}w^{\top} \Sigma w
        return (np.dot(weights, self.mean_t) - self.uncertainty_aversion*np.sqrt(np.dot(weights,np.dot(self.omega_t,weights))) - (self.risk_aversion/2)*np.sqrt(np.dot(weights,np.dot(self.cov_t,weights))) ) * c

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True) -> torch.Tensor:

        K = returns.shape[1]
        T = returns.shape[0]

        # if s-estimator, things will be a little different
        if (self.mean_estimator == "sest") and (self.covariance_estimator == "sest"):
            self.mean_t, self.cov_t = self.SEstimator(returns=returns)
        elif (self.mean_estimator == "mle"):
            self.mean_t = self.MLEMean(returns)
        elif (self.mean_estimator == "cbb") or (self.mean_estimator == "nobb") or (self.mean_estimator == "sb"):
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
        elif (self.covariance_estimator == "cbb") or (self.covariance_estimator == "nobb") or (self.covariance_estimator == "sb"):
            self.cov_t = self.DependentBootstrapCovariance(returns=returns,
                                                           boot_method=self.covariance_estimator,
                                                           Bsize=50,
                                                           rep=1000)
        elif self.covariance_estimator == "rbb":
            self.cov_t = self.DepenBootstrapCovariance(returns=returns,
                                                  boot_method=self.covariance_estimator,
                                                  Bsize= 50,
                                                  rep = 1000,
                                                  max_p= 50,
                                                  max_q= 50)
        elif self.covariance_estimator == "sest":
            pass
        else:
            raise NotImplementedError
        
        self.estimated_means.append(self.mean_t[None, :])
        self.estimated_covs.append(self.cov_t)
        
        # uncertainty set estimator
        if self.omega_estimator == "mle":
            self.omega_t = self.MLEUncertainty(T, self.cov_t)
        else:
            raise NotImplementedError
        
        # uncertainty (\kappa) aversion estimator
        if self.uncertainty_aversion_estimator == "yin-etal-2022":
            sharpe_ratios = compute_summary_statistics(returns)
            self.uncertainty_aversion = sharpe_ratios["Sharpe"] / 2 
        elif self.uncertainty_aversion_estimator == "ceria-stubbs-2006":
            # Cummulative chi-square function  
            eta = 0.95 # Ceria-Stubbs do not comment on the value they used, hence I am setting 95%
            self.uncertainty_aversion = chi2.ppf(1 - eta, df = K)
        else:
            raise NotImplementedError
        
        if long_only:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1} # allocate exactly all your assets (no leverage)
            ]
            bounds = [(0, 1) for _ in range(K)] # long-only

            w0 = np.random.uniform(0, 1, size=K)
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # "market-neutral" portfolio
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}, # allocate exactly all your assets (no leverage)
            ]
            bounds = [(-1, 1) for _ in range(K)]

            w0 = np.random.uniform(-1, 1, size=K)

        # Perform the optimization
        opt_output = opt.minimize(self.objective, w0, constraints=constraints, bounds=bounds, method='SLSQP')
        wt = torch.tensor(np.array(opt_output.x)).T.repeat(num_timesteps_out, 1)

        return wt
    
    def forward_analytic(self,
                         returns: torch.Tensor,
                         num_timesteps_out: int) -> torch.Tensor:
        pass