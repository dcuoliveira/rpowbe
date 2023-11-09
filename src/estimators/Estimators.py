import torch
import math
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize

from estimators.DependentBootstrapSampling import DependentBootstrapSampling

class Estimators:
    """
    This class implements the estimators for all the unknown quantites we have on the optimization problems.

    """
    def __init__(self) -> None:
        pass

    def MLEMean(self,
                returns: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the Maximum Likelihood estimtes of the mean of the returns.

        Args:
            returns (torch.tensor): returns tensor.
        
        Returns:
            mean_t (torch.tensor): MLE estimates for the mean of the returns.
        """
        mean_t = torch.mean(returns, axis = 0)

        return mean_t
    
    def MLECovariance(self,
                      returns: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the Maximum Likelihood estimtes of the covariance of the returns.

        Args:
            returns (torch.tensor): returns tensor.

        Returns:
            cov_t (torch.tensor): MLE estimates for the covariance of the returns.
        """
        
        cov_t = torch.cov(returns.T,correction = 0)

        return cov_t
    
    def MLEUncertainty(self,
                       T: float,
                       cov_t: torch.Tensor) -> torch.Tensor:
        """
        Method to compute the Maximum Likelihood estimtes of the uncertainty of the returns estimates.
        This method is used for the Robust Portfolio Optimization problem.

        Args:
            T (float): number of time steps.
            cov_t (torch.tensor): covariance tensor.

        Returns:
            omega_t (torch.tensor): MLE estimates for the uncertainty of the returns estimates.
        """
        
        omega_t = torch.zeros_like(cov_t)
        cov_t_diag = torch.diagonal(cov_t, 0)/T
        omega_t.diagonal().copy_(cov_t_diag)

        return omega_t
    
    def DependentBootstrapMean(self,
                               returns: torch.Tensor,
                               boot_method: str = "cbb",
                               Bsize: int = 50,
                               rep: int = 1000,
                               max_p: int = 50) -> torch.Tensor:
        """ 
        Method to compute the bootstrap mean of the returns.
        
        Args:
            returns (torch.tensor): returns tensor.
            boot_method (str): bootstrap method name to build the block set. For example, "cbb".
            Bsize (int): block size to create the block set.
            rep (int): number of bootstrap samplings to get.
            max_p (int): maximum order of the AR(p) part of the ARIMA model. Only used when boot_method = "model-based".
            max_q (int): maximum order of the MA(q) part of the ARIMA model. Only used when boot_method = "model-based".

        Returns:
            mean (torch.tensor): dependent bootstrap estimates for the mean of the returns.
        """
        
        sampler = DependentBootstrapSampling(time_series=returns,
                                             boot_method=boot_method,
                                             Bsize=Bsize,
                                             max_p=max_p)
        
        if boot_method != "sb":
            list_means = list()
            for _ in range(rep):
                boot_returns = sampler.sample()
                boot_mean = self.MLEMean(boot_returns)
                list_means.append(boot_mean)

            # compute the overall bootstrap sample mean
            smeans = torch.vstack(list_means)
            mean = torch.mean(smeans, axis=0)
        else:
            boot_returns = sampler.sample()
            mean = self.MLEMean(boot_returns)

        return mean
    
    def DependentBootstrapCovariance(self,
                                     returns: torch.Tensor,
                                     boot_method: str = "cbb",
                                     Bsize: int = 50,
                                     rep: int = 1000,
                                     max_p: int = 50) -> torch.Tensor:
        """
        Method to compute the bootstrap covariance of the returns.

        Args:
            returns (torch.tensor): returns tensor.
            boot_method (str): bootstrap method name to build the block set. For example, "cbb".
            Bsize (int): block size to create the block set.
            rep (int): number of bootstrap samplings to get.
            max_p (int): maximum order of the AR(p) part of the ARIMA model. Only used when boot_method = "model-based".
            max_q (int): maximum order of the MA(q) part of the ARIMA model. Only used when boot_method = "model-based".

        Returns:
            cov (torch.tensor): dependent bootstrap estimates for the covariance of the returns.
        """
        
        sampler = DependentBootstrapSampling(time_series=returns,
                                             boot_method=boot_method,
                                             Bsize=Bsize,
                                             max_p=max_p)
        
        if boot_method != "sb":
            list_covs = list()
            for _ in range(rep):
                boot_returns = sampler.sample()
                list_covs.append(self.MLECovariance(boot_returns))
             
             # compute the overall bootstrap sample mean
            scov = torch.stack(list_covs)
            mean_scov = torch.mean(scov, axis = 0)
        else:
            boot_returns = sampler.sample()

            mean_scov = self.MLECovariance(boot_returns)

        return mean_scov
    
    # returns both mean and covariance
    def DependentBootstrapMean_Covariance(self,
                                     returns: torch.Tensor,
                                     boot_method: str = "cbb",
                                     Bsize: int = 50,
                                     rep: int = 1000,
                                     max_p: int = 50) -> torch.Tensor:
        """
        Method to compute the bootstrap mean and covariance of the returns.

        Args:
            returns (torch.tensor): returns tensor.
            boot_method (str): bootstrap method name to build the block set. For example, "cbb".
            Bsize (int): block size to create the block set.
            rep (int): number of bootstrap samplings to get.
            max_p (int): maximum order of the AR(p) part of the ARIMA model. Only used when boot_method = "model-based".
            max_q (int): maximum order of the MA(q) part of the ARIMA model. Only used when boot_method = "model-based".

        Returns: a list of pairs containig:
            mean (torch.tensor): dependent bootstrap estimates for the mean of the returns.
            cov (torch.tensor): dependent bootstrap estimates for the covariance of the returns.
        """
        
        sampler = DependentBootstrapSampling(time_series=returns,
                                             boot_method=boot_method,
                                             Bsize=Bsize,
                                             max_p=max_p)
        
        list_mean_covs = list()
        for _ in range(rep):
            boot_returns = sampler.sample()
            list_mean_covs.append((self.MLEMean(boot_returns), self.MLECovariance(boot_returns)))
            
        # return the list of mean and covariance matrices
        return list_mean_covs
    
    def SEstimator(self, returns: torch.Tensor,
                   boot_method: str = "nobb",
                   Bsize: int = 50,
                   max_p: int = 50):
        """
        This function implements the S-estimate of the mean and covariance of the returns.

        """

        def objective(cov_t: torch.Tensor):
            return np.linalg.det(cov_t)
        
        def mahalanobis_constraint(mu_t: torch.Tensor, cov_t: torch.Tensor, t: float=2.0):
            return torch.mean(torch.tensor([mahalanobis(x, mu_t, cov_t) for x in self.boot_returns])) - t
                
        # get bootstrap samples
        sampler = DependentBootstrapSampling(time_series=returns,
                                             boot_method=boot_method,
                                             Bsize=Bsize,
                                             max_p=max_p)
        self.boot_returns = sampler.sample()
        n, d = self.boot_returns.shape
        
        constraints = ({'type': 'ineq', 'fun': mahalanobis_constraint})

        # bounds for shape parameters (non-negative)
        shape_bounds = [(0, None) for _ in range(d)]

        # set up bounds for optimization
        bounds = [(None, None)] * d + shape_bounds

        # initial guess for parameters (mean and cov matrix)
        initial_params = [torch.mean(self.boot_returns, axis=0), torch.eye(d)]
        
        # solve the optimization problem
        opt_output = minimize(objective, initial_params, method='COBYLA', constraints=constraints, bounds=bounds)   

        mean_t, cov_t = opt_output.x     

        return mean_t, cov_t



# FROM HERE S-ESTIMATOR

def SEstimator_phi(self,d):
    if math.abs(d) >= 1:
        return 0
    else:
        return (d - 2*d**3 + d**5)
    

def SEstimator_vau(self,d):
    return d*self.SEstimator_phi(d)

def SEstimator_w(self,d):
    self.SEstimator_phi(d)/d
    
# compute the Sestimator mean
def SEstimator_mean(self,
                       returns: torch.Tensor,
                       sample_mean: torch.Tensor,
                       sample_covariance: torch.Tensor) -> torch.Tensor:
    N = returns.shape[0]
    isample_covariance = torch.linalg.inv(sample_covariance)
    Smean = torch.zeros_like(sample_mean)
    tot_w = torch.zeros(1)
    for i in range(N):
        x_i = returns[i,:]
        dist = torch.matmul(x_i,torch.matmul(isample_covariance,x_i)) 
        w = dist.apply_(self.SEstimator_w)
        Smean = Smean + w*x_i
        tot_w = tot_w + w
    #
    return (Smean/tot_w)

# compute the SEstimator covariance

def SEstimator_Covariance(self,
                       returns: torch.Tensor,
                       sample_mean: torch.Tensor,
                       sample_covariance: torch.Tensor) -> torch.Tensor:
    N = returns.shape[0]
    P = returns.shape[1]
    # S-estimator mean
    SEst_mean = self.SEstimator_mean(returns,sample_mean,sample_covariance)
    centered_returns = (returns - SEst_mean)
    isample_covariance = torch.linalg.inv(sample_covariance)
    Sest_cov = torch.zeros_like(sample_covariance)
    tot_vau = torch.zeros(1)
    for i in range(N):
        x_i = centered_returns[i,:]
        dist = torch.matmul(returns[i,:],torch.matmul(isample_covariance,returns[i,:])) 
        M = torch.matmul(x_i,x_i.T) 
        w = dist.apply_(self.SEstimator_vau)
        vau = dist.apply_(self.SEs)
        Sest_cov = Sest_cov + w*M
        tot_vau = tot_vau + vau
    #
    return (P*Sest_cov/tot_vau)


#
def SEstimator_Mean_Covariance(self,
                               returns: torch.Tensor,
                               boot_method: str = "cbb",
                               Bsize: int = 50,
                               rep: int = 1000,
                               max_p: int = 50) -> torch.Tensor:
        """
        Method to compute the bootstrap  mean and covariance sestimator of the returns.

        Args:
            returns (torch.tensor): returns tensor.
            boot_method (str): bootstrap method name to build the block set. For example, "cbb".
            Bsize (int): block size to create the block set.
            rep (int): number of bootstrap samplings to get.
            max_p (int): maximum order of the AR(p) part of the ARIMA model. Only used when boot_method = "model-based".
            max_q (int): maximum order of the MA(q) part of the ARIMA model. Only used when boot_method = "model-based".

        Returns: a list of pairs containig:
            mean (torch.tensor): dependent bootstrap estimates for the mean of the returns.
            cov (torch.tensor): dependent bootstrap estimates for the covariance of the returns.
        """
        
        sampler = DependentBootstrapSampling(time_series=returns,
                                             boot_method=boot_method,
                                             Bsize=Bsize,
                                             max_p=max_p)
        
        # sample mean
        sample_mean = self.MLEMean(boot_returns)
        # sample
        sample_covariance = self.MLECovariance(boot_returns)
        #
        list_mean_covs = list()
        for _ in range(rep):
            boot_returns = sampler.sample()
            Sest_mean = self.SEstimator_mean(returns,sample_mean,sample_covariance) 
            Sest_cov = self.SEstimator_Covariance(returns,sample_mean,sample_covariance)
            list_mean_covs.append((Sest_mean, Sest_cov))
        #
        return list_mean_covs