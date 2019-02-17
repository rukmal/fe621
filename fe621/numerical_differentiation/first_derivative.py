from typing import Callable


def firstDerivative(f: Callable, x: float, h: float=1e-7) -> float:
    """Function to numerically approximate the first derivative about a point
    `x`, given a function `f(x)` which takes a single float as its argument.
    This function uses the central finite difference method, computing the slope
    of a nearby secant line passing through the points `(x - h)` and `(x + h)`.
    
    Arguments:
        f {Callable} -- Objective function who's derivative is to be computed.
        x {float} -- Point about which the derivative is computed.
    
    Keyword Arguments:
        h {float} -- Step size (default: {1e-7}).
    
    Returns:
        float -- Approximation of the first derivative of `f` about point `x`.
    """

    return (f(x + h) - f(x - h)) / (2 * h)
