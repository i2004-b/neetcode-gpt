import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        
        # X is 2D of n x m
        # weights is 2D of m x 1
        # Result will be n x 1

        # Need dot product of feature vector and weight vector
        pred = X @ weights

        return np.round(pred, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        

        # MSE is the mean of the square of the pred - actual
        mse = np.mean((model_prediction - ground_truth)**2)
        return np.round(mse, 5)
