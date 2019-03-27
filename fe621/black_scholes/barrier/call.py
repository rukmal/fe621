from .util import AnalyticalUtil

from scipy.stats import norm
import numpy as np


def callUpAndIn(S: float, H: float, volatility: float, ttm: float,
                 K: float, rf: float, dividend: float=0) -> float:
    """Analytical formula to compute the value of an up and in Barrier option.
    
    See formula 5.1 in http://bit.ly/2JHoVbQ for more.

    Arguments:
        S {float} -- Current price.
        H {float} -- Barrier price.
        volatility {float} -- Volatility of the underlying.
        ttm {float} -- Time to maturity (in years).
        K {float} -- Strike price.
        rf {float} -- Risk-free rate (annualized).
    
    Keyword Arguments:
        dividend {float} -- Dividend yield (default: {0}).
    
    Returns:
        float -- Analytical value of up and in call option.
    """

    util = AnalyticalUtil(volatility=volatility, ttm=ttm, rf=rf,
                          dividend=dividend)

    return np.power((H / S), 2 * util.nu / np.power(volatility, 2)) *\
           (util.pBS(np.power(H, 2) / S, K) - util.pBS(np.power(H, 2) / S, H) +\
           ((H - K) * np.exp(-1 * rf * ttm) * norm.cdf(-1 * util.dBS(H, S)))) +\
           util.cBS(S, H) + ((H - K) * np.exp(-1 * rf * ttm) * norm.cdf(
           util.dBS(S, H)))


def callUpAndOut(S: float, H: float, volatility: float, ttm: float, K: float,
                 rf: float, dividend: float=0) -> float:
    """Analytical formula to compute the value of an up and out barrier call
    option.

    See formula 5.2 in http://bit.ly/2JHoVbQ for more.
    
    Arguments:
        S {float} -- Current price.
        H {float} -- Barrier price.
        volatility {float} -- Volatility of the underlying.
        ttm {float} -- Time tp maturity (in years).
        K {float} -- Strike price.
        rf {float} -- Risk-free rate (annualized).
    
    Keyword Arguments:
        dividend {float} -- Dividend yield (default: {0}).
    
    Returns:
        float -- Analytical price of up and out barrier call option.
    """

    util = AnalyticalUtil(volatility=volatility, ttm=ttm, rf=rf,
                          dividend=dividend)

    return util.cBS(S, K) - util.cBS(S, H) - ((H - K) * np.exp(-1 * rf * ttm)\
        * norm.cdf(util.dBS(S, H))) - (np.power(H / S, 2 * util.nu / np.power(
        volatility, 2)) * (util.cBS(np.power(H, 2) / S, K) - util.cBS(np.power(
        H, 2) / S, H) - ((H - K) * np.exp(-1 * rf * ttm)) * norm.cdf(util.dBS(
        H, S))))
