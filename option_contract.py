from dataclasses import dataclass


@dataclass
class EuropeanOption:
    strike: float
    maturity_days: int
    option_type: str = "call"

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity_days <= 0:
            raise ValueError("maturity_days must be positive")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")