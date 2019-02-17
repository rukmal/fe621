from typing import Callable


def secondDerivative(f: Callable, x: float, h: float=1e-7) -> float:
    """Function to numerically approximate the second derivative about a point
    `x`, given a function `f(x)` which takes a single float as its argument.
    This function uses the central finite difference method, computing the slope
    of a nearby secant curve passing through the points
    `(x - h)`, `x`, and `(x - h)`.
    
    Arguments:
        f {Callable} -- Objective function who's second derivative is computed.
        x {float} -- Point about which the second derivative is computed.
    
    Keyword Arguments:
        h {float} -- Step size (default: {1e-7}).
    
    Returns:
        float -- Approxmation of the second derivative of `f` about point `x`.
    """

    return (f(x + h) - (2 * f(x)) + f(x - h)) / (h ** 2)
