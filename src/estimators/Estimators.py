import torch
from estimators.DependentBootstrapSampling import DependentBootstrapSampling

class Estimators:
    def __init__(self) -> None:
        pass

    def MLEMean(self,
                returns: torch.Tensor) -> torch.Tensor:
        
        mean_t = torch.mean(returns, axis=0)

        return mean_t
    
    def MLECovariance(self,
                      returns: torch.Tensor) -> torch.Tensor:
        
        T = returns.shape[0]
        mean_t = self.MLEMean(returns)
        cov_t = torch.matmul((returns - mean_t).T, (returns - mean_t)) / T

        return cov_t
    
    def MLEUncertainty(self,
                       T: float,
                       cov_t: torch.Tensor) -> torch.Tensor:
        
        omega_t = torch.zeros_like(cov_t)
        cov_t_diag = torch.diagonal(cov_t, 0)/T
        omega_t.diagonal().copy_(cov_t_diag)

        return omega_t
    
    # boot_method: bootstrap method to use
    # Bsize: Block size to use (not necessary for model-based)
    # rep: number of bootstrap samplings to get

    # max_P and max_Q only used when boot_method = "model-based"
    def DependentBootstrapMean(self,
                               returns: torch.Tensor,
                               boot_method: str = "cbb",
                               Bsize: int = 50,
                               rep: int = 1000,
                               max_P: int = 50,
                               max_Q: int = 50) -> torch.Tensor:
        
        sampler = DependentBootstrapSampling(time_series=returns,
                                             boot_method=boot_method,
                                             Bsize=Bsize,
                                             max_P=max_P,
                                             max_Q=max_Q)
        
        list_means = list()
        for _ in range(rep):
            boot_returns = sampler.sample()
            boot_mean = self.MLEMean(boot_returns)
            list_means.append(boot_mean)

        # compute the overall bootstrap sample mean
        smeans = torch.vstack(list_means)
        mean = torch.mean(smeans,axis = 0)
        return mean
    
    # max_P and max_Q only used when boot_method = "model-based"
    def DependentBootstrapCovariance(self,
                                     returns: torch.Tensor,
                                     boot_method: str = "cbb",
                                     Bsize: int = 50,
                                     rep: int = 1000,
                                     max_P: int = 50,
                                     max_Q: int = 50) -> torch.Tensor:
        
        sampler = DependentBootstrapSampling(time_series=returns,
                                             boot_method=boot_method,
                                             Bsize=Bsize,
                                             max_P=max_P,
                                             max_Q=max_Q)
        
        list_covs = list()
        for _ in range(rep):
            boot_returns = sampler.sample()
            list_covs.append(self.MLECovariance(boot_returns))

        # compute the overall bootstrap sample mean
        scov = torch.stack(list_covs)
        mean_scov = torch.mean(scov,axis = 0)
        return mean_scov