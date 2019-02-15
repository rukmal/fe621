from context import fe621

import numpy as np
import pandas as pd


def convergenceIterationLimit():
    # Objective function
    def f(x: float) -> float:
        return np.where(x == 0.0, 1.0, np.sin(x) / x)

    # Setting target tolerance level for termination
    epsilon = 1e-3

    # Using Trapezoidal rule
    trapezoidal_result = fe621.numerical_integration.convergenceApproximation(
        f=f,
        rule=fe621.numerical_integration.trapezoidalRule,
        epsilon=epsilon
    )

    # Using Simpson's rule
    simpsons_result = fe621.numerical_integration.convergenceApproximation(
        f=f,
        rule=fe621.numerical_integration.simpsonsRule,
        epsilon=epsilon
    )

    # Building DataFrame of results for output
    results = pd.DataFrame(np.abs(np.array([trapezoidal_result,
                                            simpsons_result])))
    
    # Setting row and column names
    results.columns = ['Estimated Area', 'Iterations']
    results.index = ['Trapezoidal Rule', 'Simpson\'s Rule']

    # Saving to CSV
    results.to_csv('Homework 1/bin/numerical_integration/convergence.csv',
                   header=True, index=True, float_format='%.8e')


if __name__ == '__main__':
    # Part 3 - Convergence Analysis
    convergenceIterationLimit()
