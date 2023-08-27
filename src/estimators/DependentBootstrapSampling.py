
import torch
import math
import random
import numpy as np
from pmdarima.arima import auto_arima
# Class for Bootstrap Sampling
# This class contains the different ways to generate time series samplings using bootstrap

class DependentBootstrapSampling:

    def __init__(self,
                 time_series: torch.Tensor,
                 boot_method: str = "cbb",
                 Bsize: int = 100,
                 max_p: int = 10,
                 max_q: int = 10) -> None:
        """
        This class contains the different ways to generate dependent bootstrap samples.

        Args:
            time_series (torch.Tensor): time series array
            boot_method (str): bootstrap method name to build the block set. For example, "cbb".
            Bsize (int): block size to create the block set.
            max_p (int): maximum order of the AR(p) part of the ARIMA model. Only used when boot_method = "model-based".
            max_q (int): maximum order of the MA(q) part of the ARIMA model. Only used when boot_method = "model-based".

        Returns:
            None
        """
        super().__init__()
        self.time_series = time_series
        self.Bsize = Bsize # not used when "boot_method" is "mbb".
        self.boot_method = boot_method
        self.Blocks = None # set of blocks
        self.Models = None # list of ARIMA models, only used when "boot_method" is "mbb".
        self.errors = None # np.array of errors, only used when "boot_method" is "mbb".
        self.Ps = None # list of best order parameter (P) of ARIMA model corresponding to each model
        if self.boot_method != "mbb": # if not "mbb" then it is model based
            self.create_blocks()
        else:
            self.create_ARIMA_models(max_p = max_p, max_q = max_q)

    def sample(self) -> torch.Tensor:
        """"
        Method to generate a sampling according to the method.

        Returns:
            sampled_data (torch.Tensor): sampled data
        """
        if self.boot_method == "cbb":
            
            N = self.time_series.shape[1]
            b = int(math.ceil(N/self.Bsize))
            selected_blocks = random.choices(self.Blocks, k = b)

            sampled_data = torch.hstack(selected_blocks)

            return sampled_data[:,:N]
        elif self.boot_method == "mbb":
            
            N = self.time_series.shape[1]
            M = self.time_series.shape[0]
            
            sampled_data = list()
            for i in range(M):
                
                model = self.Models[i]
                P = self.Ps[i]
                boot_errors = random.choices(self.errors[i],N - P)
                boot_time_series = self.time_series[i,:P]
                
                for j in range(P,N):
                    pred_X =  model.predict_in_sample(boot_time_series[(j - P):j]) + boot_errors[j - P]
                    boot_time_series = torch.hstack((boot_time_series,pred_X))
                
                sampled_data.append(boot_time_series)
                print(boot_time_series.shape)
                break
            
            sampled_data = torch.vstack(sampled_data)
            return sampled_data
    
    def create_blocks(self) -> None:
        """
        Method to create the block sets if "boot_method" specifies a block bootstrap method.

        Returns:
            (list): list of blocks
        """

        if self.boot_method == "cbb":
            self.Blocks = self.create_circular_blocks()
        else:
            self.Blocks = None
    
    def create_circular_blocks(self) -> list:
        """
        Method to create the circular block sets.

        Returns:
            Block_sets (list): list of blocks
        """

        N = self.time_series.shape[1]
        dtime_series = torch.hstack((self.time_series.clone().detach(),self.time_series[:,:(self.Bsize + 1)].clone().detach()))
        b = int(math.ceil(N/self.Bsize))

        Block_sets = list()
        for i in range(N):
            j = i + self.Bsize
            Block = dtime_series[:,i:j]
            Block_sets.append(Block)
        
        return Block_sets
    
    def create_ARIMA_models(self,
                            max_p: int,
                            max_q: int,
                            verbose: bool=False) -> None:
        """
        Method to create the ARIMA models if "boot_method" is "mbb".

        Args:
            max_p (int): maximum order of the AR(p) part of the ARIMA model.
            max_q (int): maximum order of the MA(q) part of the ARIMA model.
            verbose (bool): whether to print the progress or not.
        
        Returns:
            None
        """

        N = self.time_series.shape[1] 
        M = self.time_series.shape[0]
        
        errors = list()
        Models = list()
        Ps = list()
        # apply auto_arima for each time series
        for i in range(M):
            time_series_data = self.time_series[i,:]
            model = auto_arima(time_series_data,
                               start_p=1,
                               start_q=1,
                               test='adf',
                               max_p=max_p, max_q=max_q,
                               m=1,             
                               d=0,          
                               seasonal=False,   
                               start_P=0, 
                               D=None, 
                               trace=False,
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)
            Models.append(model)
            P, D, Q = model.order
            Ps.append(P)
            # calculate the errors
            ierrors = list()
            for j in range(0,N-P):
                error = time_series_data[j + P] - model.predict_in_sample(time_series_data[j:(j + P)])
                ierrors.append(error)
            # center ierrors
            ierrors = torch.hstack(ierrors)
            ierrors = ierrors - torch.mean(ierrors)
            errors.append(ierrors)
        
        # save
        self.Models = Models
        self.errors = errors
        self.Ps = Ps
        
        if verbose:
            print("finished")

        self.sample()
        