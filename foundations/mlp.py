import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        
        # Not told the number of layers --> figure out from the length
        layers = len(weights)

        # Want to apply relu to all layers except the last one
        for j in range(layers):
            z = x @ weights[j] + biases[j]

            if j != layers - 1:
                x = np.maximum(0.0, z)
            else:
                x = z

        return x

