import torch

class Strategy:
    """
    This class implements the Strategies given the returns and the predicted return with some Portofolio Optimization 
    Will return the strategy to use for for each asset.
    Paper: https://arxiv.org/pdf/1904.04912.pdf
    """
    def __init__(self,
                 time_series: torch.Tensor,
                 pred_returns: torch.Tensor
                ) -> None:
        #
        super().__init__()
        self.time_series = time_series
        self.pred_returns = pred_returns

    def Moskowitz(self) -> torch.Tensor:
        """
        Moskowitz Method to compute the strategy for asset allocation.
        
        Returns:
            position_sizing (torch.tensor): Moskowitz method for Position sizing with +1 or -1
        """
        # consider the last year
        N = self.time_series.shape[0]
        L = 252
        # obtain the mean of the returns of all assets in the past L days
        year_returns = torch.mean(self.time_series[max(0,(N - L)):N,:],axis = 0)
        # calculate the position sizing
        position_sizing = torch.sign(year_returns)

        return position_sizing
    
    # BAZ method implementation
    def Baz(self,
            L: int,
            S: int) -> torch.Tensor:
        """
        Baz Method to compute the strategy for asset allocation.
        Args:
        L: long time scale (24,48,96)
        S: short time scale (8,16,32)
        Returns:
            position_sizing (torch.tensor): Moskowitz method for Position sizing with values between +1 or -1
        """
        N = self.time_series.shape[0]
        MACD = self.EWMA(S) - self.EWMA(L) 
        # standard deviation of the last 63 days
        STD = torch.std(self.time_series_window[(N - 63):N,:],axis = 0)
        STD_year = torch.std(self.time_series_window[(N - 252):N,:],axis = 0)
        # calculate Q
        Q = MACD/STD
        #
        Y = Q/STD_year
        # compute positions
        position_sizing = (Y*torch.exp(-(Y*Y)/4.0))/0.89

        return position_sizing
    
    # Evaluate Strategy (TSOM: Time series momentum)
    def evaluate_strategy(self,
                          real_returns: torch.Tensor,
                          #predicted_returns: torch.Tensor,
                          method: str = "Moskowitz",
                          L: int = 24,
                          S: int = 8) -> torch.Tensor:
        # calculate strategy
        position_sizing = None
        if method == "Moskowitz":
            position_sizing = self.Moskowitz()
        else: # Baz method selected
            position_sizing = self.Baz(L,S)
        # VOLATILITY
        sigma_TGT = 0.15
        sigma_t = self.EWSD(60)
        # compute TSOM
        return_TSOM = torch.mean(sigma_TGT*((position_sizing*real_returns)/sigma_t))
        # return value
        return return_TSOM


    # Exponential Weighted Average of all assets of the last S days
    # TO DO: correct when S is larger than the number of time series points!
    def EWMA(self,
             S: int) -> torch.Tensor:
        N = self.time_series.shape[0]
        alpha = (S-1)/S
        time_series_window = self.time_series[(N - S):N,:]
        weights = torch.ones(1, S)
        for idx in range(S-2,-1,-1):
            weights[0,idx] = (1-alpha)*weights[0,idx + 1]
        #
        weights = alpha*weights
        #
        return (torch.matmul(weights, time_series_window))
    
    # Exponential Weighted Standard Deviation of all assets of the last S days
    # TO DO: correct when S is larger than the number of time series points!
    def EWSD(self,
             S: int) -> torch.Tensor:
        N = self.time_series.shape[0]
        alpha = (S-1)/S
        time_series_window = self.time_series[(N - S):N,:]
        weights = torch.ones(S,1)
        for idx in range(S-2,-1,-1):
            weights[idx,0] = (1-alpha)*weights[idx + 1,0]
        #
        wtime_series_window = time_series_window*weights
        mean_wtime_series_window = torch.sum(wtime_series_window,axis = 0)
        sd_wtime_series_window =  (wtime_series_window - mean_wtime_series_window)*(wtime_series_window - mean_wtime_series_window)
        #
        return (torch.mean(sd_wtime_series_window, axis = 0))