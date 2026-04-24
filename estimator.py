import numpy as np
from scipy.optimize import minimize

from parameters import HNParameters
from likelihood import LikelihoodEvaluator


class HNEstimator:
    def __init__(self, rate: float):
        self.rate = rate
        self.likelihood = LikelihoodEvaluator(rate)

    def _unpack(self, x: np.ndarray) -> HNParameters:
        return HNParameters.from_vector(x)

    def _objective(self, x: np.ndarray, returns: np.ndarray) -> float:
        try:
            params = self._unpack(x)

            if not params.is_stationary():
                return 1e10

            return self.likelihood.negative_log_likelihood(returns, params)
        except Exception:
            return 1e10

    def fit(self, returns: np.ndarray, x0: np.ndarray):
        result = minimize(
            fun=self._objective,
            x0=np.asarray(x0, dtype=float),
            args=(np.asarray(returns, dtype=float),),
            method="L-BFGS-B",
            bounds=[
                (None, None),   # lambda_
                (1e-12, None),  # omega
                (1e-12, None),  # alpha
                (1e-12, None),  # beta
                (None, None),   # gamma
            ],
        )
        return result