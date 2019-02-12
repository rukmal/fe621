from .util import computeD1D2

from scipy.stats import norm

import numpy as np


def callDelta(current: float, volatility: float, ttm: float, strike: float,
              rf: float) -> float:
    """Function to compute the Delta of a call option using the Black-Scholes
    formula.
    
    Arguments:
        current {float} -- Current price of the underlying asset.
        volatility {float} -- Volatility of the underlying asset price.
        ttm {float} -- Time to expiration (in years).
        strike {float} -- Strike price of the option contract.
        rf {float} -- Risk-free rate (annual).
    
    Returns:
        float -- Delta of a European Call Option contract.
    """

    d1, d2 = computeD1D2(current, volatility, ttm, strike, rf)

    return np.cdf(d1)


def callGamma(current: float, volatility: float, ttm: float, strike: float,
              rf: float) -> float:
    """Function to compute the Gamma of a Call option using the Black-Scholes
    formula.
    
    Arguments:
        current {float} -- Current price of the underlying asset.
        volatility {float} -- Volatility of the underlying asset price.
        ttm {float} -- Time to expiration (in years).
        strike {float} -- Strike price of the option contract.
        rf {float} -- Risk-free rate (annual).
    
    Returns:
        float -- Delta of a European Call Opton Option contract.
    """

    d1, d2 = computeD1D2(current, volatility, ttm, strike)

    return norm.pdf(d1) * (1 / (current * volatility * np.sqrt(ttm)))


def vega(current: float, volatility: float, ttm: float, strike: float,
         rf: float) -> float:
    """Function to compute the Vega of an option using the Black-Scholes formula.
    
    Arguments:
        current {float} -- Current price of the underlying asset.
        volatility {float} -- Volatility of the underlying asset price.
        ttm {float} -- Time to expiration (in years).
        strike {float} -- Strike price of the option contract.
        rf {float} -- Risk-free rate (annual).
    
    Returns:
        float -- Vega of a European Option contract.
    """

    d1, d2 = computeD1D2(current, volatility, ttm, strike, rf)

    return current * np.sqrt(ttm) * norm.pdf(d1)
