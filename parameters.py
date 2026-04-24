from dataclasses import dataclass


@dataclass
class HNParameters:
    lambda_: float
    omega: float
    alpha: float
    beta: float
    gamma: float

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")

    def persistence(self) -> float:
        return self.beta + self.alpha * self.gamma**2

    def is_stationary(self) -> bool:
        return self.persistence() < 1.0

    def to_vector(self):
        return [self.lambda_, self.omega, self.alpha, self.beta, self.gamma]

    @classmethod
    def from_vector(cls, x):
        return cls(
            lambda_=x[0],
            omega=x[1],
            alpha=x[2],
            beta=x[3],
            gamma=x[4],
        )

    def to_risk_neutral(self) -> "RiskNeutralHNParameters":
        gamma_q = self.gamma + self.lambda_ + 0.5
        return RiskNeutralHNParameters(
            omega=self.omega,
            alpha=self.alpha,
            beta=self.beta,
            lambda_q=-0.5,
            gamma_q=gamma_q,
        )


@dataclass
class RiskNeutralHNParameters:
    omega: float
    alpha: float
    beta: float
    lambda_q: float
    gamma_q: float

    def to_vector(self):
        return [self.omega, self.alpha, self.beta, self.lambda_q, self.gamma_q]