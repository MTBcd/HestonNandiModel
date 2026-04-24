import numpy as np

from option_contract import EuropeanOption
from state import HNState
from parameters import RiskNeutralHNParameters


class MonteCarloHNPricer:
    def __init__(self, rn_params: RiskNeutralHNParameters, rate: float, n_paths: int = 10000, seed: int | None = None):
        self.params = rn_params
        self.rate = rate
        self.n_paths = n_paths
        self.seed = seed

    def price(self, option: EuropeanOption, state: HNState) -> float:
        rng = np.random.default_rng(self.seed)

        S = np.full(self.n_paths, state.spot, dtype=float)
        h = np.full(self.n_paths, state.variance, dtype=float)

        for _ in range(option.maturity_days):
            z = rng.standard_normal(self.n_paths)

            S = S * np.exp(self.rate - 0.5 * h + np.sqrt(h) * z)
            h = (
                self.params.omega
                + self.params.beta * h
                + self.params.alpha * (z - self.params.gamma_q * np.sqrt(h))**2
            )

        if option.option_type == "call":
            payoff = np.maximum(S - option.strike, 0.0)
        else:
            payoff = np.maximum(option.strike - S, 0.0)

        return float(np.exp(-self.rate * option.maturity_days) * np.mean(payoff))
    

    