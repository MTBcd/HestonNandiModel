import numpy as np
from parameters import HNParameters
from variance_filter import VarianceFilter


class LikelihoodEvaluator:
    def __init__(self, rate: float):
        self.rate = rate  # daily rate

    def log_likelihood(self, returns: np.ndarray, parameters: HNParameters) -> float:
        vf = VarianceFilter(parameters, self.rate)
        h = vf.filter_path(returns)

        residual = returns - self.rate - parameters.lambda_ * h

        if np.any(h <= 0) or not np.all(np.isfinite(h)):
            raise ValueError("variance path contains invalid values")

        ll_terms = -0.5 * (
            np.log(2.0 * np.pi)
            + np.log(h)
            + (residual**2) / h
        )

        return float(np.sum(ll_terms))

    def negative_log_likelihood(self, returns: np.ndarray, parameters: HNParameters) -> float:
        return -self.log_likelihood(returns, parameters)