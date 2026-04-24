import numpy as np
from parameters import HNParameters


class VarianceFilter:
    def __init__(self, parameters: HNParameters, rate: float):
        self.parameters = parameters
        self.rate = rate  # daily rate

    def initial_variance(self) -> float:
        # The initial variance can be calculated as omega / (1 - beta - alpha * gamma^2) which is the unconditional variance of the process
        p = self.parameters
        denominator = 1.0 - (p.beta + p.alpha * p.gamma**2)
        if denominator <= 0:
            raise ValueError("non-stationary or invalid parameters")
        return p.omega / denominator

    def standardized_residual(self, r_t: float, h_t: float) -> float:
        # Standardized residual z_t = (r_t - rate - lambda * h_t) / sqrt(h_t)
        p = self.parameters
        return (r_t - self.rate - p.lambda_ * h_t) / np.sqrt(h_t)

    def next_variance(self, r_t: float, h_t: float) -> float:
        # Next variance h_{t+1} = omega + beta * h_t + alpha * (z_t - gamma * sqrt(h_t))^2
        p = self.parameters
        z_t = self.standardized_residual(r_t, h_t)
        return p.omega + p.beta * h_t + p.alpha * (z_t - p.gamma * np.sqrt(h_t))**2

    def filter_path(self, returns: np.ndarray) -> np.ndarray:
        # Use the initial and recursive formulas to compute the variance path given the returns
        returns = np.asarray(returns, dtype=float)

        if len(returns) == 0:
            raise ValueError("returns cannot be empty")
        # Initialize the variance array
        h = np.zeros(len(returns), dtype=float)
        h[0] = self.initial_variance()
        # Recursively compute the variance path
        for t in range(len(returns) - 1):
            h[t + 1] = self.next_variance(returns[t], h[t])

        return h

