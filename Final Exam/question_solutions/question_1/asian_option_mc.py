# Specific simulation functions for the Monte Carlo simulation pricing of
# Asian (i.e. average value) Options.
from context import fe621

from typing import Callable
import numpy as np

def sim_func_geometric(x: np.array, gbm: Callable, computePayoffPV: Callable,
    current: float) -> float:
    """Monte Carlo simulation function to compute the geometric-average price
    of an Asian (i.e. average value) Call option. This function is intended to
    be used with the fe621 package Monte Carlo skeleton.
    
    Arguments:
        x {np.array} -- Random numbers (for a single simulation).
        gbm {Callable} -- Function to model the GBM of the asset.
        computePayoffPV {Callable} -- Function to compute the present value of
                                      the temrinal value of the option.
        current {float} -- Current (i.e. initial) asset price.
    
    Returns:
        float -- Asian Call option price.
    """

    # Computing price deltas
    log_price_deltas = gbm(x)

    # Computing log price path
    # Note: Appending log of current (i.e. initial) price to this array too
    log_price_path = np.cumsum(np.append(np.log(current), log_price_deltas))

    # Computing geometric average of the price path
    # (simply the arithmetic average of the log prices, raised to e)
    avg_log_price = np.exp(np.mean(log_price_path))

    # Computing PV of the terminal payoff; returning
    return np.maximum(computePayoffPV(avg_log_price), 0)


def sim_func_arithmetic(x: np.array, gbm: Callable, computePayoffPV: Callable,
    current: float) -> float:
    """Monte Carlo simulation function to compute the arithmetic-average price
    of an Asian (i.e. average value) Call option. This function is intended to
    be used with the fe621 package Monte Carlo skeleton.
    
    Arguments:
        x {np.array} -- Random numbers (for a single simulation).
        gbm {Callable} -- Function to model the GBM of the asset.
        computePayoffPV {Callable} -- Function to compute the present value of
                                      the temrinal value of the option.
        current {float} -- Current (i.e. initial) asset price.
    
    Returns:
        float -- Asian Call option price.
    """

    # Computing price path
    price_path = current * np.exp(np.cumsum(gbm(x)))
    # Appending the initial (i.e. current) price
    price_path = np.append(current, price_path)

    # Computing the arithmetic average of the price path
    avg_price = np.mean(price_path)

    # Computing PV of terminal payoff; returning
    return np.maximum(computePayoffPV(avg_price), 0)

