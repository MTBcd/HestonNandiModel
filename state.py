from dataclasses import dataclass
from market_data import MarketData
from parameters import HNParameters


@dataclass
class HNState:
    spot: float
    variance: float

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.variance <= 0:
            raise ValueError("variance must be positive")
        




