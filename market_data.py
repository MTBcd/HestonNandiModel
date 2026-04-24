from dataclasses import dataclass
import numpy as np

@dataclass
class MarketData:
    returns: np.ndarray  # List of historical returns
    spot: float  # Initial stock price
    rate: float   # Risk-free rate

    def __post_init__(self):
        if self.spot <= 0:
            raise ValueError("Spot price must be positive.")
        if self.rate < 0:
            raise ValueError("Risk-free rate must be non-negative.")
        if len(self.returns) == 0:
            raise ValueError("Return series cannot be empty.")
        if not np.all(np.isfinite(self.returns)):
            raise ValueError("Returns must be finite numbers.")

    @property
    def n_obs(self) -> int:
        return len(self.returns)

    @property
    def daily_rate(self) -> float:
        if self.rate < 0.0001:
            return self.rate  # Assume it's already a daily rate if it's very small
        return self.rate / 365.25

