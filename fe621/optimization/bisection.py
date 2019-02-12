from typing import Callable
import numpy as np


def bisectionSolver(f: Callable, a: float, b: float,
                    tol: float=10e-6) -> float:
    """Bisection method solver, implemented using recursion.
    
    Arguments:
        f {Callable} -- Function to be optimized.
        a {float} -- Lower bound.
        b {float} -- Upper bound.
    
    Keyword Arguments:
        tol {float} -- Solution tolerance (default: {10e-6}).
    
    Raises:
        Exception -- Raised if no solution is found.
    
    Returns:
        float -- Solution to the function s.t. f(x) = 0.
    """

    # Compute midpoint
    mid = (a + b) / 2

    # Check if estimate is within tolerance
    if (b - a) < tol:
        return mid
    
    # Evaluate function at midpoint
    f_mid = f(mid)

    # Check position of estimate, move point and re-evaluate
    if (f(a) * f_mid) < 0:
        return bisectionSolver(f=f, a=a, b=mid)
    elif (f(b) * f_mid) < 0:
        return bisectionSolver(f=f, a=mid, b=b)
    else:
        raise Exception("No solution found.")

