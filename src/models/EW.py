import torch

class EW:
    def __init__(self) -> None:
        pass

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        return torch.ones(returns.shape[0], returns.shape[1]) / returns.shape[1]