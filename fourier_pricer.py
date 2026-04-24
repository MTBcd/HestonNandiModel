import numpy as np
from scipy.integrate import quad

from option_contract import EuropeanOption
from state import HNState
from affine_recursion import AffineRecursionEngine


class FourierHNPricer:
    def __init__(self, recursion_engine: AffineRecursionEngine, rate: float):
        self.engine = recursion_engine
        self.rate = rate  # daily rate

    def integrand(self, u: float, option: EuropeanOption, state: HNState) -> float:
        phi_1 = 1.0 + 1j * u
        phi_2 = 1j * u

        f1 = self.engine.transform(phi_1, state, option.maturity_days)
        f2 = self.engine.transform(phi_2, state, option.maturity_days)

        value = np.exp(-self.rate * option.maturity_days) * np.real(
            np.exp(-1j * u * np.log(option.strike)) * (
                f1 / (option.strike * 1j * u) - f2 / (1j * u)
            )
        )
        return float(value)

    def price(self, option: EuropeanOption, state: HNState) -> float:
        option.validate()
        state.validate()

        integral_value, _ = quad(
            lambda u: self.integrand(u, option, state),
            1e-8,
            100.0,
            limit=200,
        )

        price = (
            0.5 * (
                state.spot
                - option.strike * np.exp(-self.rate * option.maturity_days)
            )
            + (option.strike / np.pi) * integral_value
        )

        return float(price)