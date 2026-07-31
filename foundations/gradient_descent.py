class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        
        # Initialize x_old and x_new to be init
        x_new = x_old = init
        # Iterate for a certain number of times
        for _ in range(iterations):
            x_new = x_old - learning_rate * (2 * x_old)
            # Assign x_old to x_new for the next iteration
            x_old = x_new

        return round(x_new, 5)


