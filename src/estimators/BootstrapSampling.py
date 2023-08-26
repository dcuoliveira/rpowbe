
import torch
import math
import random
# Class for Bootstrap Sampling
# This class contains the different ways to generate time series samplings using bootstrap

class BootstrapSampling:

    def __init__(self,
                 time_series: torch.Tensor,
                 boot_method: str = "circular",
                 Bsize: int = 100) -> None:
        """"
        Args:
            time_series (np.array): time series array
            boot_method: bootstrap method name to build the block set. For example, "circular"
            Bsize: block size to creare the block set.
        """
        super().__init__()
        self.time_series = time_series
        self.Bsize = Bsize # not used when "boot_method" is "model-based".
        self.boot_method = boot_method
        self.Blocks = None # set of blocks
        self.Models = None # list of ARIMA models, only used when "boot_method" is "model-based".
        self.errors = None # np.array of errors, only used when "boot_method" is "model-based".
        if self.boot_method != "model-based": # if not "model-based" then it is model based
            self.create_blocks()
        else:
            self.create_ARIMA_models()

    # generates a sampling according to the method
    def sample(self) -> torch.Tensor:
        if self.boot_method != "model-based":
            N = self.time_series.shape[1]
            b = int(math.ceil(N/self.Bsize))
            selected_blocks = random.choices(self.Blocks,k = b)

            sampled_data = torch.hstack(selected_blocks)

            return sampled_data[:,:N]
        else:
            return None
    #
    def create_blocks(self) -> None:
        """"
        Method to create the block sets if "boot_method" specifies a block bootstrap method. In case that
        such variable is "model-based" then it will build the set of ARIMA models (for each time series) and
        """
        if self.boot_method == "circular":
            self.Blocks = self.create_circular_blocks()
        else:
            self.Blocks = None
    #
    def create_circular_blocks(self) -> list:
        N = self.time_series.shape[1]
        dtime_series = torch.hstack((self.time_series.clone().detach(),self.time_series[:,:(self.Bsize + 1)].clone().detach()))
        b = int(math.ceil(N/self.Bsize))

        Block_sets = list()
        for i in range(N):
            j = i + self.Bsize
            Block = dtime_series[:,i:j]
            Block_sets.append(Block)
        #
        return Block_sets
    #
    def create_ARIMA_models(self):
        return (None,None)