from typing import Callable
import numpy as np


def newtonSolver(f: Callable, f_prime: Callable, guess: float,
                 tol: float=10e-6, prev: float=0) -> float:
    """Newton method solver for 1 dimension, implemented recursively.
    
    Arguments:
        f {Callable} -- Objective function (must have zero root).
        f_prime {Callable} -- First derivative of objective with respect to
                              the decision variable.
        guess {float} -- Guess for the decision variable.
    
    Keyword Arguments:
        tol {float} -- Tolerance level (default: {10e-6}).
        prev {float} -- Guess from previous iteration (for convergence check).
    
    Returns:
        float -- Solution to the function s.t. f(x) = 0.
    """

    # Assigning current guess to x_old
    x_old = guess

    # Checking if decision variable changed by less than tolerance level
    if np.abs(x_old - prev) < tol:
        return x_old
    else:
        # Compute new estimate for x
        x_new = x_old - (f(x_old) / f_prime(x_old))
        return newtonSolver(f=f, f_prime=f_prime, guess=x_new,
                            tol=tol, prev=x_old)
