from scipy.stats import norm
from typing import Callable
import numpy as np


def monteCarloSkeleton(sim_count: int, eval_count: int, sim_func: Callable,
    sim_dimensionality: int=1) -> np.array:
    """Function to run a simple Monte Carlo simulation. This is a highly
    generalized Monte Carlo simulation skeleton, and takes in functions as
    parameters for computation functions, and final post-processing
    functionality.

    This function uses list comprehensions to improve performance.
    
    Arguments:
        sim_count {int} -- Simulation count.
        eval_count {int} -- Number of evaluations per simulation.
        sim_func {Callable} -- Function to run on the random numbers
                               (per-simulation).

    Keyword Arguments
        sim_dimensionality {int} -- Dimensionality of the simulation. Affects
                                    the shape of random normals (default: {1}).
    
    Returns:
        np.array -- Array of simulated value outputs.
    """
    
    # Simulation function
    def simulation() -> float:
        """Single simulation run. This is written as a separate function so I
        can use list comprehensions in the outer loop, giving this operation
        a significant performance bump.
        """

        # Building list of normal random numbers to apply to sim_func
        rand_Ns = norm.rvs(size=(sim_dimensionality, eval_count))
        # Applying simulated function over path
        return sim_func(rand_Ns)
    
    # Running simulations the required number of times, returning
    return np.array([simulation() for i in range(0, sim_count)])


def monteCarloStats(mc_output: np.array) -> dict:
    """Function to compute statistics on a Monte Carlo simulation output set.

    This function computes the estimate (i.e. the mean), sample standard
    deviation (i.e. std. with delta degrees of freedom = 1), and the standard
    error of the Monte Carlo simulation output array.
    
    Arguments:
        mc_output {np.array} -- Array of simulated Monte Carlo values.
    
    Returns:
        dict -- Dictionary with summary statistics.
    """

    # Empty dictionary to store output
    output = dict()

    # Estimate
    output['estimate'] = np.mean(mc_output)
    # Standard deviation (sample)
    output['standard_deviation'] = np.std(mc_output, ddof=1)
    # Standard error
    output['standard_error'] = output['standard_deviation'] / np.sqrt(
        len(mc_output))

    # Return final output
    return output
