import torch
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