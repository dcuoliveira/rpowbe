import torch
from numpy.linalg import eig

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

    def eigenvalues(self, x: torch.tensor) -> torch.tensor:
        """
        This function computes the eigenvalues of a given matrix.

        Args:
            x (torch.tensor): input matrix.
        Returns:
            torch.tensor: eigenvalues of x.
        """

        # for each cov matrix, get the maximum eigenvalue
        max_eigenvalues = torch.tensor([eig(x[i])[0].max() for i in range(len(x))])

        return max_eigenvalues

    def means(self, x: torch.tensor) -> torch.tensor:
        """
        This function computes the means of a given matrix.

        Args:
            x (torch.tensor): input matrix.
        Returns:
            torch.tensor: means of x.
        """

        means_vec = torch.mean(torch.stack(x), axis=1)

        return means_vec

    def percentile(self, x: torch.tensor, scores: torch.tensor, alpha: float=0.95) -> float:
        """
        This function computes the alpha-percentile of a given vector.

        Args:
            x (torch.tensor): input vector.
            alpha (float): percentile. Defaults to 0.95.
        Returns:
            float: alpha-percentile of x.
        """
        n = len(x)

        scores_idx = sorted(range(len(scores)), key=lambda k: scores[k])
        selected_idx = scores_idx[int(alpha * n)]
        x_selected = x[selected_idx]

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
            scores = self.eigenvalues(x)
        elif func == "means":
            scores = self.means(x)

        x_selected = self.percentile(x, scores, alpha=self.alpha)
        
        return x_selected