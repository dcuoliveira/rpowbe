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