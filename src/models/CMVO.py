import torch
import numpy as np
import scipy.optimize as opt
from utils.AutomaticCluster import AutomaticCluster

from estimators.Estimators import Estimators

class CMVO(Estimators):
    def __init__(self,
                 risk_aversion: float=1,
                 clus_method: str = "Lrw",
                 ) -> None:
        """"
        This function impements the mean-variance optimization (MVO) method proposed by Markowitz (1952).

        Args:
            risk_aversion (float): risk aversion parameter. Defaults to 0.5. 
                                   The risk aversion parameter is a scalar that controls the trade-off between risk and return.
                                   According to Ang (2014), the risk aversion parameter of a risk neutral individual ranges from 1 and 10.
            clus_method (str): "Lrw": Laplacian, "Lsym": normalized Laplacian, "SPONGE", "SPONGEsym"
            mean_estimator (str): mean estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.
            covariance_estimator (str): covariance estimator to be used. Defaults to "mle", which defines the maximum likelihood estimator.

        References:
        Markowitz, H. (1952) Portfolio Selection. The Journal of Finance.
        Ang, Andrew, (2014). Asset Management: A Systematic Approach to Factor Investing. Oxford University Press. 
        """
        super().__init__()
        
        self.risk_aversion = risk_aversion
        self.estimated_covs = list()
        self.clus_method = clus_method
        self.num_clusters = None

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True) -> torch.Tensor:

        # mean,cov, and corr estimates
        self.list_mean_covs = [(self.MLEMean(returns),self.MLECovariance(returns),torch.corrcoef(returns.t()))]
        # for each bootstraped time series
        # 1.- Compute The clustering
        # 2.- For each label, compute the mvo
        # 3.- COmpute the accumulative 
        results_boot = list()

        ## include range of values for the number of clusters
        mean,cov,corr = self.list_mean_covs[0]
        labels = self.clustering(corr)
        list_models = list()
        accum_fun = 0.0
        ulabels = set(labels)
        for lab in ulabels:
            indicator = (labels == lab)
            mean_clus = mean[indicator]
            cov_clus = cov[indicator,:]
            cov_clus = cov_clus[:,indicator]
            model_clus = self.apply_MVO(mean_clus,cov_clus,long_only=long_only)
            list_models.append(model_clus)
            accum_fun = accum_fun + model_clus.fun
            

        # now build the predictions array
        pred = np.zeros(shape = (1,returns.shape[1]))
        ulabels = list(ulabels)
        for idx in range(len(ulabels)):
            indicator = (labels == ulabels[idx])
            pred[0,indicator] = list_models[idx].x
        #
        wt = torch.tensor(np.array(pred)).T.repeat(num_timesteps_out, 1).T

        return wt
    


    # clustering using sigNet
    # automatically selects the number of clusters, like in Cucuringu's paper
    # returns the clusters it obtained
    def clustering(self,
                   corr: torch.Tensor,) -> np.array:
        # using: pip install git+https://github.com/alan-turing-institute/SigNet.git
        Ap = np.array(torch.where(corr > 0, corr, 0.))
        An = np.abs(np.array(torch.where(corr < 0, corr, 0.)))
        clus_met = AutomaticCluster((Ap, An),0.90)
        labels = clus_met.spectral_cluster_laplacian(select_clust_n="spectral", normalisation='sym_sep')
        return labels
    
    # apply MVO optimization to a given timeseries
    # return 
    def apply_MVO(self,
                  mean: torch.Tensor,
                  cov: torch.Tensor,
                  long_only: bool=True) -> any:
        K = cov.shape[0]
        w0 = None
        
        if long_only:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1} # allocate exactly all your assets (no leverage)
            ]
            bounds = [(0, 1) for _ in range(K)] # long-only

            w0 = np.random.uniform(0, 1, size=K)
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # "market-neutral" portfolio
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}, # allocate exactly all your assets (no leverage)
            ]
            bounds = [(-1, 1) for _ in range(K)]

            w0 = np.random.uniform(-1, 1, size=K)

        # define objective
        def objective(weights: torch.Tensor,
                      maximize: bool=True) -> torch.Tensor:
        
            c = -1 if maximize else 1
            
            return (np.dot(weights, mean) - ((self.risk_aversion/2) * np.sqrt(np.dot(weights, np.dot(cov, weights))) )) * c
        
        # perform the optimization
        opt_output = opt.minimize(objective, w0, constraints=constraints, bounds=bounds)
        #
        return (opt_output)

