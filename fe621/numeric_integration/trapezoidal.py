from typing import Callable
import numpy as np


def trapezoidalRule(f: Callable, N: float, start: float=-1e6,
                    stop: float=1e6) -> float:
    """Function to approximate the numeric integral of a function, f, using
    the Trapezoidal rule.
    
    Arguments:
        f {Callable} -- Function who's integral is to be estimated.
        N {int} -- Number of nodes to consider.
    
    Keyword Arguments:
        start {float} -- Starting point (default: {-1e6}).
        stop {float} -- Stopping point (default: {1e6}).
    
    Returns:
        float -- Approximation of the area under the function.
    """

    # Building values for approximation, and getting step size
    x, h = np.linspace(start=start, stop=stop, num=N, retstep=True)

    # Estimating area using trapezoidal rule, return
    return np.sum((h * f(x))) - ((h / 2) * (f(start) + f(stop)))
