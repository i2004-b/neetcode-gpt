import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        

        # Turn inputs into np array
        x = np.array(x) # Holds floats
        W1 = np.array(W1) # 2D, holds floats --> hidden * input (transpose to match)
        b1 = np.array(b1) # Holds floats
        W2 = np.array(W2) # 2D, holds floats --> output * hidden (transpose to match)
        b2 = np.array(b2) # Holds floats
        y_true = np.array(y_true)

        # Forward pass
        z1 = x @ np.transpose(W1) + b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ np.transpose(W2) + b2

        # Calculate the loss
        loss = np.mean((z2 - y_true) ** 2)

        # Use the loss for the backprop derivatives
        # Find the length of z2 (used for the derivative of the loss)
        n = len(y_true) if y_true.ndim > 0 else 1

        # Derive loss with respect to z2
        dz2 = 2 * (z2 - y_true) / n

        # Second layer: z2 = a1W2T + b2
        # What we have: dL/dz2
        # What we need: dL/dW2 and dL/db2

        # Solving for dL/dW2
        # Use intermediary --> dz2/dW2 = a1
        dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1)

        # Solving for dL/db2
        # Use intermediary --> dz2/db2 = 1
        db2 = dz2

        # Solving for dL/da1
        # Use intermediary --> dz2/da1 = W2
        da1 = dz2.reshape(1, -1) @ W2

        # Go back through ReLU
        da1 = da1.flatten()
        dz1 = da1 * (z1 > 0).astype(float)
        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
        db1 = dz1



        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }

       




