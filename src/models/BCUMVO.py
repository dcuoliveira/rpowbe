import torch
import numpy as np
import scipy.optimize as opt
from utils.Clustering import Clustering


from estimators.Estimators import Estimators

class BCUMVO(Estimators):
    def __init__(self,
                 risk_aversion: float=1,
                 mean_cov_estimator: str="mle",
                 num_boot: int = 200,
                 alpha: float = 0.95,
                 clus_method: str = "Lrw",
                 min_num_clusters: int = 2,
                 max_num_clusters: int = 10,
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
        self.mean_cov_estimator = mean_cov_estimator
        self.estimated_covs = list()
        self.clus_method = clus_method
        self.num_boot = num_boot
        self.alpha = alpha
        self.min_num_clusters = min_num_clusters
        self.max_num_clusters = max_num_clusters
        self.best_num_clusters = None # will save the best number of clusters

    def forward(self,
                returns: torch.Tensor,
                num_timesteps_out: int,
                long_only: bool=True) -> torch.Tensor:

        # mean and cov estimates
        if self.mean_cov_estimator == "mle":
            raise ValueError("MLE is not supported for this model")
        else:
            self.list_mean_covs = self.DependentBootstrapMean_Covariance_Corr(returns=returns,
                                                                         boot_method=self.mean_cov_estimator,
                                                                         Bsize=50,
                                                                         rep=self.num_boot)
        # for each bootstraped time series
        # 1.- Compute The clustering
        # 2.- For each label, compute the mvo
        # 3.- COmpute the accumulative 
        result_cluster = list() # saves all 
        for k in range(self.min_num_clusters,self.max_num_clusters + 1):
            results_boot = list()
            for idx in range(len(self.list_mean_covs)):
                mean,cov,corr = self.list_mean_covs[idx]
                labels = self.clustering(corr,k)
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
                
                # multiply by -1 because we want it maximized
                results_boot.append((-1*accum_fun,labels,list_models,ulabels))
            # obtain the percentile for each cluster number
            # now sort and take the percentile
            sorted_results_boot = sorted(results_boot, key = lambda x : x[0],reverse=False)
            # get alpha-percentile utility
            pos = int((1 - self.alpha)*(len(self.list_mean_covs) - 1))
            utility,labels, models,ulabels = sorted_results_boot[pos]
            # append the utility/labels/models/ulabels of the best alpha percentile for a fixed number of clusters
            result_cluster.append((utility,labels, models,ulabels))
        # get the number of clusters with the maximum utility
        _,labels,models,ulabels = max(result_cluster, key=lambda tup: tup[0])
        # saving the best number of clusters
        self.best_num_clusters = len(ulabels)
        # now build the predictions array
        pred = np.zeros(shape = (1,returns.shape[1]))
        ulabels = list(ulabels)
        for idx in range(len(ulabels)):
            indicator = (labels == ulabels[idx])
            pred[0,indicator] = models[idx].x
        #
        wt = torch.tensor(np.array(pred)).T.repeat(num_timesteps_out, 1).T

        return wt

    # clustering using sigNet
    # automatically selects the number of clusters, like in Cucuringu's paper
    # returns the clusters it obtained
    def clustering(self,
                   corr: torch.Tensor,
                   k: int) -> np.array:
        # using: pip install git+https://github.com/alan-turing-institute/SigNet.git
        Ap = np.array(torch.where(corr > 0, corr, 0.))
        An = np.abs(np.array(torch.where(corr < 0, corr, 0.)))
        clus_met = Clustering((Ap, An))
        labels = clus_met.spectral_cluster_laplacian(k, normalisation='sym_sep')
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
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # the weights sum to one
            ]
            bounds = [(0, 1) for _ in range(K)]

            w0 = np.random.uniform(0, 1, size=K)
        else:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 0},  # the weights sum to zero
                {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1},  # the weights sum to zero
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

