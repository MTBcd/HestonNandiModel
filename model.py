from market_data import MarketData
from parameters import HNParameters
from state import HNState
from variance_filter import VarianceFilter


class HestonNandiModel:
    def __init__(self, parameters: HNParameters, market_data: MarketData):
        self.parameters = parameters
        self.market_data = market_data

    def filter_variance(self):
        vf = VarianceFilter(self.parameters, self.market_data.rate)
        return vf.filter_path(self.market_data.returns)

    def current_state(self) -> HNState:
        h_path = self.filter_variance()
        return HNState(
            spot=self.market_data.spot,
            variance=h_path[-1],
        )

    def risk_neutral_parameters(self):
        return self.parameters.to_risk_neutral()
    
