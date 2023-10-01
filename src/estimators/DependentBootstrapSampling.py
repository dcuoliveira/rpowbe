
import torch
import math
import random
import numpy as np
# Import Statsmodels for VAR
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
# Class for Bootstrap Sampling
# This class contains the different ways to generate time series samplings using bootstrap

class DependentBootstrapSampling:

    def __init__(self,
                 time_series: torch.Tensor,
                 boot_method: str = "cbb",
                 Bsize: int = 100,
                 max_p: int = 4) -> None:
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
        self.Bsize = Bsize # not used when "boot_method" is "rbb".
        self.boot_method = boot_method
        self.Blocks = None # set of blocks
        self.Model = None # list of ARIMA models, only used when "boot_method" is "rbb".
        self.residuals = None # np.array of errors, only used when "boot_method" is "rbb".
        self.P = None # best order parameter (P) of VAR model corresponding to each model
        if self.boot_method != "rbb": # if not "rbb" then it is model based
            self.create_blocks()
        else:
            self.create_VAR_model(max_p = max_p)

    def sample(self) -> torch.Tensor:
        """"
        Method to generate a sampling according to the method.
        Implemented methods are:
        - Non-overlapping Block Bootstrap (nobb)
        - Circular Block Bootstrap (cbb)
        - Stationary Bootstrap (sb)
        - Model-based / Residual-based Bootstrap (rbb)

        Returns:
            sampled_data (torch.Tensor): sampled data
        """
        
        if self.boot_method == "cbb":

            N = self.time_series.shape[0]
            b = int(math.ceil(N / self.Bsize))
            selected_blocks = random.choices(self.Blocks, k = b)

            sampled_data = torch.vstack(selected_blocks)
            return sampled_data[:N, :]
    
        elif self.boot_method == "nobb":

            N = self.time_series.shape[0]
            b = int(math.ceil(N / self.Bsize))
            selected_blocks = random.choices(self.Blocks, k = b)

            sampled_data = torch.vstack(selected_blocks)

            return sampled_data[:N, :]

        elif self.boot_method == "rbb":
            
            N = self.time_series.shape[0]
            M = self.time_series.shape[1]
            # first P are equal
            sampled_data = self.time_series[0:self.P,:].clone()
            # build array for residuals
            random_residuals = np.zeros((0,self.residuals.shape[0]))
            for j in range(M):
                random_col = np.array(random.choices(self.residuals[:,j],k = self.residuals.shape[0]))
                random_residuals = np.vstack([random_residuals,random_col])
            #
            random_residuals = random_residuals.T
            #
            for j in range(N - self.P):
                new_observation = self.Model.forecast(sampled_data[j:(j + self.P),:],1)
                new_observation = new_observation + random_residuals[j,:]
                sampled_data = np.vstack([sampled_data,new_observation])

            return torch.Tensor(sampled_data)

        elif self.boot_method == "sb":
            sampled_data = torch.vstack(self.Blocks)
            
            return torch.Tensor(sampled_data)
    
    def create_blocks(self) -> None:
        """
        Method to create the block sets if "boot_method" specifies a block bootstrap method.

        Returns:
            (list): list of blocks
        """

        if self.boot_method == "cbb":
            self.Blocks = self.create_circular_blocks()
        elif self.boot_method == "nobb":
            self.Blocks = self.create_non_overlapping_blocks()
        elif self.boot_method == "sb":
            self.Blocks = self.create_stationary_blocks()
        else:
            self.Blocks = None

    def create_stationary_blocks(self) -> list:
        """
        Method to create the block sets to be used in the stationary bootstrap model.

        Returns:
            Block_sets (list): list of blocks
        """

        N = self.time_series.shape[0]

        Block_sets = list()
        Ls = list()
        Is = list()

        total = 0
        i = 0
        while total < N:

            # write me a line of code to generate a random integer number between 1 and N
            I = random.randint(1, N)
            L = np.random.geometric(p=0.5, size=1)[0]

            Block = self.time_series[I:(I + L), :]
            Block_sets.append(Block)

            Ls.append(L)
            Is.append(I)

            total += L
            i += 1

        return Block_sets
    
    def create_non_overlapping_blocks(self) -> list:
        """
        Method to create the non-overlapping block sets.

        Returns:
            Block_sets (list): list of blocks
        """

        N = self.time_series.shape[0]

        Block_sets = list()
        for i in range(0, N, self.Bsize):
            Block = self.time_series[i:min((i + self.Bsize),N),:]
            Block_sets.append(Block)

        return Block_sets
    
    def create_circular_blocks(self) -> list:
        """
        Method to create the circular block sets.

        Returns:
            Block_sets (list): list of blocks
        """

        N = self.time_series.shape[0]
        dtime_series = torch.vstack([self.time_series.clone().detach(),self.time_series[:self.Bsize,:].clone().detach()])

        Block_sets = list()
        for i in range(N):
            j = i + self.Bsize
            Block = dtime_series[i:j,:]
            Block_sets.append(Block)
        
        return Block_sets
    
    # TODO: solve problem when the best parameters are (0,0,0) -> which means that is completely random
    def create_VAR_model(self,
                            max_p: int = 10,
                            verbose: bool=False) -> None:
        """
        Method to create the ARIMA models if "boot_method" is "rbb".

        Args:
            max_p (int): maximum order of the VAR(p) part of the VAR model.
            verbose (bool): whether to print the progress or not.
        
        Returns:
            None
        """

        N = self.time_series.shape[0] 
        M = self.time_series.shape[1]
        
        # FIND ORDER
        np_time_series = self.time_series.numpy().copy()
        self.P = self.find_VAR_order(np_time_series,max_p)
        model = VAR(np_time_series)
        self.Model = model.fit(self.P)
        # compute the residuals
        residuals = np.zeros((0,M))
        for i in range(self.P,N):
            new_point = self.Model.forecast(np_time_series[(i-self.P):i,:],1)
            residuals = np.vstack([residuals,new_point])

        # put mean of the residuals to zero
        col_means = residuals.sum(axis = 0)
        self.residuals = residuals - col_means
        if verbose:
            print("finished")

        #self.sample()
    
    #  find the best order by using AIC
    def find_VAR_order(self,
                       np_time_series: np.array,
                       max_p: int) -> int:

        model = VAR(np_time_series)
        # use AIC because BIC returns 0
        results = model.fit(maxlags = max_p, ic='aic')
        #
        return (results.k_ar)