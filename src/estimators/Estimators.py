import torch
from estimators.BootstrapSampling import BootstrapSampling

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

    def BootstrapMean(self,
                      returns: torch.Tensor,
                      boot_method: str = "circular",
                      Bsize: int = 50,
                      rep = 1000) -> torch.Tensor:
        sampler = BootstrapSampling(time_series = returns,boot_method = boot_method,Bsize = Bsize)
        list_means = list()
        for _ in range(rep):
            boot_returns = sampler.sample()
            boot_mean = self.MLEMean(boot_returns)
            list_means.append(boot_mean)
        # compute the overall bootstrap sample mean
        smeans = torch.vstack(list_means)
        mean = torch.mean(smeans,axis = 0)
        return mean
    
    def BootstrapCovariance(self,
                      returns: torch.Tensor,
                      boot_method: str = "circular",
                      Bsize: int = 50,
                      rep = 1000) -> torch.Tensor:
        sampler = BootstrapSampling(time_series = returns,boot_method = boot_method,Bsize = Bsize)
        list_covs = list()
        for _ in range(rep):
            boot_returns = sampler.sample()
            list_covs.append(self.MLECovariance(boot_returns))
        # compute the overall bootstrap sample mean
        scov = torch.stack(list_covs)
        mean_scov = torch.mean(scov,axis = 0)
        return mean_scov