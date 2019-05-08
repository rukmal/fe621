from ..monte_carlo import monteCarloSkeleton, monteCarloStats

import numpy as np


def blackScholes(current: float, volatility: float, ttm: float, strike: float,
                 rf: float, dividend: float, sim_count: int, eval_count: int,
                 opt_type: str='C') -> dict:
    """Function to model the price of a European Option, under the Black-Scholes
    pricing model heuristic, using a Monte-Carlo simulation.

    This function simulates a simple Geometric Brownian Motion (GBM) of the
    underlying asset price, before computing the terminal contract value for a
    given number of simulated paths. Then, Monte Carlo simulation statistics are
    computed for each of the simulations, and a dict of results is returned.
    
    Arguments:
        current {float} -- Current price of the underlying asset.
        volatility {float} -- Volatility of the underlying asset price.
        ttm {float} -- Time to expiration (in years).
        strike {float} -- Strike price of the option contract.
        rf {float} -- Risk-free rate (annual).
        dividend {float} -- Dividend yield (annual).
        sim_count {int} -- Number of paths to simulate.
        eval_count {int} -- Number of evaluations per path simulation.
    
    Keyword Arguments:
        opt_type {str} -- Option type; must be 'C' or 'P' (default: {'C'}).
    
    Raises:
        ValueError: Raised if `opt_type` is not 'C' or 'P'.
    
    Returns:
        dict -- Formatted dictionary of Monte Carlo simulation results.
    """

    # Verify option type choice
    if opt_type not in ['C', 'P']:
        raise ValueError('Incorrect option type; must be "C" or "P".')

    # Computing delta t
    dt = ttm / eval_count
    # Computing intitial value
    init_val = np.log(current)
    # Computing nudt
    nudt = (rf - dividend - (np.power(volatility, 2) / 2)) * dt

    # Defining lambda function to model Geometric Brownian Motion (GBM)
    gbm = lambda x: nudt + (volatility * np.sqrt(dt) * x)

    # Defining simulation function
    def sim_func(x: np.array) -> float:
        if (opt_type == 'C'):
            # Call option
            return np.exp(-1 * rf * ttm) * \
                np.maximum(np.exp(init_val + np.sum(gbm(x))) - strike, 0)
        else:
            # Put option
            return np.exp(-1 * rf * ttm) * \
                np.maximum(strike - np.exp(init_val + np.sum(gbm(x))), 0)

    # Running simulation
    mc_output = monteCarloSkeleton(sim_count=sim_count,
                                   eval_count=eval_count,
                                   sim_func=sim_func)

    # Computing and returning sample statistics
    return monteCarloStats(mc_output=mc_output)
