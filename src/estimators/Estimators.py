import torch

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