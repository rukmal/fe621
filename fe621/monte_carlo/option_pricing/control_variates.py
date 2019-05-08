from ..monte_carlo import monteCarloSkeleton, monteCarloStats
from ...black_scholes.greeks import callDelta, putDelta

import numpy as np


def deltaCVBlackScholes(current: float, volatility: float, ttm: float,
                      strike: float, rf: float, dividend: float, sim_count: int, eval_count: int, beta1: float, opt_type: str='C') -> dict:
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
        beta {float} -- Beta coefficient for the delta hedge.
    
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
    # Computing nudt
    nudt = (rf - dividend - (np.power(volatility, 2) / 2)) * dt
    # Delta bias correction
    erddt = np.exp((rf - dividend) * dt)

    # Building vector of ttms (for option delta evaluation)
    # Note: This starts from timestep 1, to timestep eval_count.
    # Note: This is the time to maturity, the order must be flipped to match
    #       the simulated asset prices (at the first sim price
    #       it is ((ttm - dt), (ttm - 2*dt), ...)
    ttm_vec = np.flip(np.arange(start=dt, stop=(ttm + dt), step=dt))

    # Defining lambda function to model underlying Geometric Brownian Motion,
    # and Delta-based control variate
    gbm = lambda x: nudt + (volatility * np.sqrt(dt) * x)

    # Defining simulation function
    def sim_func(x: np.array) -> float:
        # Underlying price path
        st = np.cumprod(np.exp(gbm(x))) * current

        if (opt_type == 'C'):
            # Call option
            # Delta computation
            delta = callDelta(current=st,
                              volatility=volatility,
                              ttm=ttm_vec,
                              strike=strike,
                              rf=rf,
                              dividend=dividend)

            # Terminal payoff computation (future value)
            terminal_payoff = np.maximum(st[-1] - strike, 0)
        else:
            # Put option
            # Delta computation
            delta = putDelta(current=st,
                             volatility=volatility,
                             ttm=ttm_vec,
                             strike=strike,
                             rf=rf,
                             dividend=dividend)
            # Terminal payoff computation (future value)
            terminal_payoff = np.maximum(strike - st[-1], 0)

        # Control variate computation
        cv = np.sum(delta[:-1] * (st[1:] - (st[:-1] * erddt)))

        # Adjusting estimate by control variate; returning present value
        return np.exp(-1 * rf * ttm) * (terminal_payoff + (cv * beta1))

    # Runnig simulation
    mc_output = monteCarloSkeleton(sim_count=sim_count,
                                   eval_count=eval_count,
                                   sim_func=sim_func)

    # Computing and returning sample statistics
    return monteCarloStats(mc_output=mc_output)
