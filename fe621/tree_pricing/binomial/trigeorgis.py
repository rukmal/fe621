from ..general_tree import GeneralTree

import numpy as np


class Trigeorgis(GeneralTree):
    def __init__(self, current: float, strike: float, ttm: float, rf: float,
                 volatility: float, steps: int=1, type=str, style=str):

        self.deltaX = 0
        self.jumpProbability = 0
        
        super().__init__(current=current, strike=strike, ttm=ttm,
                           rf=rf, steps=steps)

    def getDeltaX(self) -> np.array:
        return np.array([self._current_val, 0, self._current_val])
    
    def getJumpProbability(self) -> np.array:
        return self.jumpProbability
    
    def getVolatility(self) -> float:
        return 0.0

    def instrumentValueAtNode(self) -> float:
        return 0.0
    
    def instrumentValueFromChildren(self) -> float:
        return 0.0
