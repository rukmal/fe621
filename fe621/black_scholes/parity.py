import numpy as np


def call(put: float, current: float, strike: float, ttm: float,
         rf: float) -> float:
    """Function to compute the price of a European Call option contract from a
    European Put option contract price using Put-Call parity.
    
    Arguments:
        put {float} -- Price of the put option.
        current {float} -- Current price of the underlying asset.
        strike {float} -- Strike price of the option contract.
        ttm {float} -- Time to expiration (in years).
        rf {float} -- Risk-free rate (annual).
    
    Returns:
        float -- Price of a European Call Option contract.
    """

    return put + current - (strike * np.exp(-1 * rf * ttm))


def put(call: float, current: float, strike: float, ttm: float,
        rf: float) -> float:
    """Function to compute the price of a European Put option contract from a
    European Call option contract price using Put-Call parity.
    
    Arguments:
        call {float} -- Price of the call option.
        current {float} -- Current price of the underlying asset.
        strike {float} -- Strike price of the option contract.
        ttm {float} -- Time to expiration (in years).
        rf {float} -- Risk-free rate (annual).
    
    Returns:
        float -- Price of a European Put Option contract.
    """

    return call - current + (strike * np.exp(-1 * rf * ttm))
