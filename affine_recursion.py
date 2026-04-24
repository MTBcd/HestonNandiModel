import numpy as np
from parameters import RiskNeutralHNParameters
from state import HNState


class AffineRecursionEngine:
    def __init__(self, rn_params: RiskNeutralHNParameters, rate: float):
        self.params = rn_params
        self.rate = rate  # daily rate

    def terminal_conditions(self, phi: complex) -> tuple[complex, complex]:
        A = phi * self.rate
        B = self.params.lambda_q * phi + 0.5 * phi**2
        return A, B

    def backward_step(self, phi: complex, A_next: complex, B_next: complex) -> tuple[complex, complex]:
        p = self.params

        denominator = 1.0 - 2.0 * p.alpha * B_next
        if denominator == 0:
            raise ZeroDivisionError("denominator in affine recursion is zero")

        A_t = (
            A_next
            + phi * self.rate
            + p.omega * B_next
            - 0.5 * np.log(denominator)
        )

        B_t = (
            phi * (p.lambda_q + p.gamma_q)
            - 0.5 * p.gamma_q**2
            + p.beta * B_next
            + 0.5 * ((phi - p.gamma_q) ** 2) / denominator
        )

        return A_t, B_t

    def run_recursion(self, phi: complex, maturity_days: int) -> tuple[complex, complex]:
        A, B = self.terminal_conditions(phi)

        for _ in range(maturity_days - 1):
            A, B = self.backward_step(phi, A, B)

        return A, B

    def transform(self, phi: complex, state: HNState, maturity_days: int) -> complex:
        A, B = self.run_recursion(phi, maturity_days)
        return state.spot**phi * np.exp(A + B * state.variance)