from ..call import blackScholesCall
from ..put import blackScholesPut
from ..util import computeD1D2

import numpy as np


class AnalyticalUtil():
    """Helper class for analytical barrier option pricing.

    See chapter 5 in http://bit.ly/2JHoVbQ for more.
    """

    def __init__(self, volatility: float, ttm: float, rf: float,
        dividend: float=0):
        self.volatility = volatility
        self.ttm = ttm
        self.rf = rf
        self.dividend = dividend
        self.nu = self.rf - self.dividend - (np.power(self.volatility, 2) / 2)
    
    def cBS(self, current: float, strike: float) -> float:
        return blackScholesCall(current=current,
                                volatility=self.volatility,
                                ttm=self.ttm,
                                strike=strike,
                                rf=self.rf)

    def pBS(self, current: float, strike: float) -> float:
        return blackScholesPut(current=current,
                                volatility=self.volatility,
                                ttm=self.ttm,
                                strike=strike,
                                rf=self.rf)

    def dBS(self, current: float, strike: float) -> float:
        return (np.log(current / strike) + (self.nu * self.ttm)) /\
            (self.volatility * np.sqrt(self.ttm))
