import pandas as pd
import numpy as np

from market_data import MarketData
from option_contract import EuropeanOption
from parameters import HNParameters
from model import HestonNandiModel
from affine_recursion import AffineRecursionEngine
from fourier_pricer import FourierHNPricer
from monte_carlo_pricer import MonteCarloHNPricer
from estimator import HNEstimator


def main():
    VL = pd.read_csv("historique_valeurs_liquidatives-FR0011170182.csv")["VL"].values

    returns = np.log1p(np.diff(VL) / VL[:-1])

    market = MarketData(
        returns=returns,
        spot=100.0,
        rate=0.01,   # annual rate
    )

    option = EuropeanOption(
        strike=100.0,
        maturity_days=30,
        option_type="call",
    )

    # Initial guess for calibration
    x0 = np.array([0.0, 1e-6, 1e-6, 0.80, 5.0], dtype=float)

    estimator = HNEstimator(market.daily_rate)
    result = estimator.fit(market.returns, x0)

    fitted_params = HNParameters.from_vector(result.x)
    model = HestonNandiModel(fitted_params, market)

    state = model.current_state()
    rn_params = model.risk_neutral_parameters()

    engine = AffineRecursionEngine(rn_params, market.daily_rate)
    fourier_pricer = FourierHNPricer(engine, market.daily_rate)
    mc_pricer = MonteCarloHNPricer(rn_params, market.daily_rate, n_paths=100000, seed=42)

    fourier_price = fourier_pricer.price(option, state)
    mc_price = mc_pricer.price(option, state)

    print("Optimization success:", result.success)
    print("Fitted parameters:", fitted_params)
    print("Current state:", state)
    print("Fourier price:", fourier_price)
    print("Monte Carlo price:", mc_price)


if __name__ == "__main__":
    main()