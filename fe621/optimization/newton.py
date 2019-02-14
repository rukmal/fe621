from typing import Callable
import numpy as np


def newtonSolver(f: Callable, f_prime: Callable, guess: float,
                 tol: float=10e-6) -> float:
    """Newton method solver for 1 dimension, implemented recursively.
    
    Arguments:
        f {Callable} -- Objective function (must have zero root).
        f_prime {Callable} -- First derivative of objective with respect to
                              the decision variable.
        guess {float} -- Guess for the decision variable.
    
    Keyword Arguments:
        tol {float} -- Tolerance level (default: {10e-6}).
    
    Returns:
        float -- Solution to the function s.t. f(x) = 0.
    """

    x_old = guess

    if np.abs(f(x_old)) < tol:
        return x_old
    else:
        x_new = x_old - (f(x_old) / f_prime(x_old))
        return newtonSolver(f=f, f_prime=f_prime, guess=x_new, tol=tol)

