from typing import Tuple

import numpy as np


def computeD1D2(current: float, volatility: float, ttm: float, strike: float,
                rf: float) -> Tuple[float, float]:
    """Helper function to compute the risk-adjusted priors of exercising the
    option contract, and keeping the underlying asset. This is used in the
    computation of both the Call and Put options in the
    Black-Scholes-Merton framework.
    
    Arguments:
        current {float} -- Current price of the underlying asset.
        volatility {float} -- Volatility of the underlying asset price.
        ttm {float} -- Time to expiration (in years).
        strike {float} -- Strike price of the option contract.
        rf {float} -- Risk-free rate (annual).
    
    Returns:
        Tuple[float, float] -- Tuple with d1, and d2 respectively.
    """

    d1 = (np.log(current / strike) + (rf + ((volatility ** 2) / 2)) * ttm) \
        / (volatility * np.sqrt(ttm))
    d2 = d1 - (volatility * np.sqrt(ttm))
    
    return (d1, d2)
