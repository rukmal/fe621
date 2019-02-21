from typing import Callable, Tuple
import numpy as np


def convergenceApproximation(f: Callable, rule: Callable, epsilon: float=1e-3,
                             start: float=-1e6, stop: float=1e6) \
                            -> Tuple[float, int]:
    """Function to approximate the numeric integral of a function, f, using
    a given quadrature rule and a tolerance level epsilon.
    
    Arguments:
        f {Callable} -- Function for which the integral is to be estimated.
        rule {Callable} -- Function to be used to approximate area. Must take
                           positional arguments f, N, start and stop.
    
    Keyword Arguments:
        epsilon {float} -- Tolerance level (default: {1e-3}).
        start {float} -- Starting point (default: {-1e6}).
        stop {float} -- Stopping point (default: {1e6}).
    
    Returns:
        Tuple[float, int] -- Approximation of the area under the function
                             and the number of segments (area, segments).
    """

    # Flags
    area_old = 0
    area_new = 1
    N = 1

    while (np.abs(area_new - area_old) > epsilon):
        # Set new area to old area
        area_old = area_new

        # Increase N by factor of 10
        N *= 10

        # Computing area with given parameters
        area_new = rule(f=f, N=N, start=start, stop=stop)

        # Log
        print('On iteration {0} method {1} convergence {2} val {3}'.format(
            N, str(rule),
            '{:.5e}'.format(np.abs(area_new - area_old)),
            area_new))

    # Return final area and number of segments
    return (area_new, N)
