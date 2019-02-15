from typing import Callable
import numpy as np


def simpsonsRule(f: Callable, N: float, start: float=-1e6,
                 stop: float=1e6) -> float:
    """Function to approximate the numeric integral of a function, f, using
    Simpson's rule.
    
    Arguments:
        f {Callable} -- Function for which the integral is to be estimated.
        N {float} -- Number of nodes to consider.
    
    Keyword Arguments:
        start {float} -- Starting point (default: {-1e6}).
        stop {float} -- Stopping point (default: {1e6}).
    
    Returns:
        float -- Approximation of the area under the function.
    """

    # Building values for approximation, and getting step size
    x, h = np.linspace(start=start, stop=stop, num=N, retstep=True)

    # Computing midpoints
    x_mid = np.array([(x[i - 1] + x[i]) / 2 for i in range(1, N)])

    # Estimating using Simpson's rule
    area = np.sum(2 * f(x)) - (f(start) + f(stop)) + (4 * np.sum(f(x_mid)))

    # Scaling area
    area *= (h / 6)

    return area
