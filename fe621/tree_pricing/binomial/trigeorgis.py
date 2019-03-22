from ..general_tree import GeneralTree

import numpy as np


class Trigeorgis(GeneralTree):
    def __init__(self, current: float, strike: float, ttm: float, rf: float,
                 volatility: float, steps: int=1, type=str, style=str):

        self.rf = rf
        self.volatility = volatility
        self.deltaX = 0
        self.jumpProbability = 0
        
        # Computing deltaT
        deltaT = ttm / steps

        # Computing upward and downward jumps for children
        # Do this only once so it doesn't have to be recomputed each time
        # Upward additive deltaX
        self.deltaXU = np.sqrt((np.power(rf - (np.power(volatility, 2) / 2), 2)\
                               * np.power(deltaT, 2)) + (np.power(volatility,
                               2) * deltaT))
        # Down deltaX = -1 * upDeltaX
        self.deltaXD = -1 * self.deltaXU

        super().__init__(current=current, strike=strike, ttm=ttm,
                           rf=rf, steps=steps)

    def getChildren(self) -> np.array:
        # Computing up and downward child additive values (mid is 0)
        upValue = self._current_val + self.deltaXU
        downValue = self._current_val + self.deltaXD

        return np.array([upValue, 0, downValue])
    
    def getJumpProbability(self) -> np.array:
        return self.jumpProbability
    
    def getVolatility(self) -> float:
        return 0.0

    def instrumentValueAtNode(self) -> float:
        return 0.0
    
    def instrumentValueFromChildren(self) -> float:
        return 0.0
