import torch
from numpy.linalg import eig
import numpy as np
import pandas as pd

class Functionals:
    def __init__(self, alpha: float=0.95) -> None:
        """
        This class implments a collection of functionals to rank vectors and matrices.
        The rankk procedure consists of the following steps:
            1. Compute score for each vector/matrix. (e.g. eigenvalues, means, etc)
            3. Compute the percentile of the score.

        Args:
            alpha (float): percentile. Defaults to 0.95.
        """

        self.alpha = alpha

    def eigenvalues(self, x: list) -> torch.tensor:
        """
        This function computes the eigenvalues of a given matrix.

        Args:
            x (torch.tensor): input matrix.
        Returns:
            torch.tensor: eigenvalues of x.
        """

        # for each cov matrix, get the maximum eigenvalue
        max_eigenvalues = torch.tensor([eig(x[i])[0].real.max() for i in range(len(x))])

        return max_eigenvalues

    def means(self, x: list) -> torch.tensor:
        """
        This function computes the means of a given matrix.

        Args:
            x (list): input matrix.
        Returns:
            torch.tensor: means of x.
        """

        if len(x) == 1:
            return x[0]
        
        means_vec = torch.mean(torch.stack(x), axis=1)

        return means_vec
    
    def utility(self, x: list) -> torch.tensor:
        """
        This function computes the utility of a given portfolio.
        
        Args:
            x (list): input list with mean (1st position) and cov (2nd position).
            risk_aversion (float): risk aversion parameter. Defaults to 1.
        Returns:
            torch.tensor: utilities of x.
        """

        wt = torch.Tensor(np.random.uniform(-1, 1, size = self.K))

        # compute utilities for all bootstraps
        utilities = list()
        for idx in range(len(x)):
            mean_i, cov_i = x[idx]
            utility = torch.matmul(wt, mean_i) - self.risk_aversion*torch.matmul(wt, torch.matmul(cov_i, wt))
            utilities.append(utility.item())

        return utilities

    def percentile(self, x: torch.tensor, scores: torch.tensor, alpha: float=0.95) -> float:
        """
        This function computes the alpha-percentile of a given vector.

        Args:
            x (torch.tensor): input vector.
            alpha (float): percentile. Defaults to 0.95.
        Returns:
            float: alpha-percentile of x.
        """

        if len(x) == 1:
            return x[0]

        # n is typically the numbre of bootstrap samples
        n = len(x)
        percentile_idx = int(alpha * n) - 1

        scores_estimates = pd.DataFrame({"score": scores, "estimates": x}).sort_values(by="score", ascending=True).reset_index(drop=True)
        x_selected = scores_estimates.iloc[percentile_idx]['estimates']

        return x_selected
    
    def maximum(self, x: torch.tensor, scores: torch.tensor) -> float:
        """
        This function computes the maximum of a given vector.

        Args:
            x (torch.tensor): input vector.
        Returns:
            float: maximum of x.
        """

        if len(x) == 1:
            return x[0]

        scores_estimates = pd.DataFrame({"score": scores, "estimates": x}).sort_values(by="score", ascending=True).reset_index(drop=True)
        x_selected = scores_estimates.iloc[-1]['estimates']

        return x_selected
    
    def minimum(self, x: torch.tensor, scores: torch.tensor) -> float:
        """
        This function computes the minimum of a given vector.

        Args:
            x (torch.tensor): input vector.
        Returns:
            float: minimum of x.
        """

        if len(x) == 1:
            return x[0]

        scores_estimates = pd.DataFrame({"score": scores, "estimates": x}).sort_values(by="score", ascending=True).reset_index(drop=True)
        x_selected = scores_estimates.iloc[0]['estimates']

        return x_selected

    def apply_functional(self, x: torch.tensor, func: str="eigenvalues") -> torch.tensor:
        """
        This function applies a functional to a given matrix.

        Args:
            x (torch.tensor): input matrix.
            func (str): functional to be applied. Defaults to "eigenvalues".
        Returns:
            torch.tensor: output of the functional.
        """

        if func == "eigenvalues":
            self.scores = self.eigenvalues(x)
        elif func == "means":
            self.scores = self.means(x)
        elif func == "utility":
            self.scores = self.utility(x)
        else:
            raise ValueError("Functional not implemented.")

        if self.alpha == -1:
            x_selected = self.minimum(x, self.scores)
        elif self.alpha == 1:
            x_selected = self.maximum(x, self.scores)
        else:
            x_selected = self.percentile(x, self.scores, alpha=self.alpha)
        
        return x_selected
    
    
    def find_utility_position(self, utilities, utility_value):
        """
        Find the position of the utility tensor corresponding to the given final utility valuer.

        Parameters
        ----------
        utilities : list
            List of utilities.
        utility_value : float
            Final utility value.
        
        Returns
        -------
        int
            Position of the utility tensor corresponding to the given final utility valuer.

        """
        
        for i, utility in enumerate(utilities):

            check = (torch.sum(utility == utility_value) == len(utility))

            if check:
                return i
            
        return None