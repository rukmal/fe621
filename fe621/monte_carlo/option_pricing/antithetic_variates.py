from ..monte_carlo import monteCarloSkeleton, monteCarloStats

import numpy as np


def blackScholes(current: float, volatility: float, ttm: float, strike: float,
                 rf: float, dividend: float, sim_count: int, eval_count: int,
                 opt_type: str='C', **kwargs) -> dict:
    """Function to model the price of a European Option, under the
    Black-Scholes pricing model heuristic, using an antithetic variates method
    variance-reduced Monte-Carlo simulation.

    This function simulates two perfectly negatively correlated simple Geometric
    Brownian Motion (GBM) processes of the underlying asset price, before
    computing the terminal contract value for a given number of simulated paths,
    as the arithmetic average of the payouts of each of the two GBMs.

    Then, Monte Carlo simulation statistics are computed for each of the
    simulations, and a dict of results is returned.
    
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

    # Verify option choice
    if opt_type not in ['C', 'P']:
        raise ValueError('Incorrect option type; must be "C" or "P".')
    
    # Computing delta t
    dt = ttm / eval_count
    # Computing initial value
    init_val = np.log(current)
    # Computing nudt
    nudt = (rf - dividend - (np.power(volatility, 2) / 2)) * dt

    # Defining lambda functions to model Geometric Brownian Motion (GBM);
    # for both assets (negatively correlated)
    gbm1 = lambda x: nudt + (volatility * np.sqrt(dt) * x)
    gbm2 = lambda x: nudt + (volatility * np.sqrt(dt) * (-1 * x))

    # Defining simulation function
    def sim_func(x: np.array) -> float:
        if (opt_type == 'C'):
            # Call option
            return np.exp(-1 * rf * ttm) * 0.5 * (
                np.maximum(np.exp(init_val + np.sum(gbm1(x))) - strike, 0) +
                np.maximum(np.exp(init_val + np.sum(gbm2(x))) - strike, 0))
        else:
            # Put option
            return np.exp(-1 * rf * ttm) * 0.5 * (
                np.maximum(strike - np.exp(init_val + np.sum(gbm1(x))), 0) +
                np.maximum(strike - np.exp(init_val + np.sum(gbm2(x))), 0))
    
    # Running simulation
    mc_output = monteCarloSkeleton(sim_count=sim_count,
                                   eval_count=eval_count,
                                   sim_func=sim_func)

    # Computing and returning sample statistics
    return monteCarloStats(mc_output=mc_output)
