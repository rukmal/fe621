from ...util.config import cfg

from scipy.stats import norm
import numpy as np

def call(current: float, volatility: float, ttm: float, strike: float,
        rf: float, days_in_year: int=cfg.days_in_year) -> float:
    """Function to compute the value of an Asian (i.e. average value) Call
    Option under the Black-Scholes model world heuristic. The option is
    parameterized using the underlying asset price, volatility,
    time to expiration, strike price, and risk-free rate.

    Note that this formulation utilizes a daily average computation window,
    scaling the time to expiration by the number of days in a year.
    
    Arguments:
        current {float} -- Current price of the underlying asset.
        volatility {float} -- Volatility of the underlying asset price.
        ttm {float} -- Time to expiration (in years).
        strike {float} -- Strike price of the option contract.
        rf {float} -- Risk-free rate (annual).
    
    Keyword Arguments:
        days_in_year {int} -- Number of days in a trading year
                              (default: {cfg.days_in_year}).
    
    Returns:
        float -- Analytically computed price of the Asian call option.
    """

    # Total number of trading days
    N = days_in_year * ttm

    # Computing adjusted option metadata for analytical formula
    adj_vol = volatility * np.sqrt(((2 * N) + 1) / (6 * (N + 1))) # Adjusted vol
    rho = 0.5 * (rf - (np.power(volatility, 2) / 2) + np.power(adj_vol, 2))

    # Money-ness probabilities
    d1 = (1 / (np.sqrt(ttm) * adj_vol)) * (np.log(current / strike) + ((rho + \
        (0.5 * np.power(adj_vol, 2))) * ttm))
    d2 = (1 / (np.sqrt(ttm) * adj_vol)) * (np.log(current / strike) + ((rho - \
        (0.5 * np.power(adj_vol, 2))) * ttm))
    
    # Computing final price with formula; return
    return np.exp(-1 * rf * ttm) * ((current * np.exp(rho * ttm) * norm.cdf(d1))
        - (strike * norm.cdf(d2)))
