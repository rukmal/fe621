from context import fe621

import numpy as np
import pandas as pd


def arbitraryFunctionSegmentAnalysis():
    """Function to analyze number of segments required for an arbitrary function
    to converge under the Trapezoidal and Simpson's quadrature rules.
    """

    # Defining objective function
    def g(x: float) -> float:
        return 1 + np.exp(-1 * np.power(x, 2)) * np.cos(8 * np.power(x, 2/3))
    
    # Setting target tolerance level for termination
    epsilon = 1e-4

    # Setting start and stop limits
    start = 0
    stop = 2

    # Trapezoidal rule
    trapezoidal_result = fe621.numerical_integration.convergenceApproximation(
        f=g,
        rule=fe621.numerical_integration.trapezoidalRule,
        start=start,
        stop=stop,
        epsilon=epsilon
    )

    # Simpson's rule
    simpsons_result = fe621.numerical_integration.convergenceApproximation(
        f=g,
        rule=fe621.numerical_integration.simpsonsRule,
        start=start,
        stop=stop,
        epsilon=epsilon
    )

    # Building DataFrame of results for output
    results = pd.DataFrame(np.abs(np.array([trapezoidal_result,
                                            simpsons_result])))
    
    # Setting row and column names
    results.columns = ['Estimated Area', 'Segments']
    results.index = ['Trapezoidal Rule', 'Simpson\'s Rule']

    # Saving to CSV
    results.to_csv('Homework 1/bin/numerical_integration/arb_convergence.csv',
                   header=True, index=True, float_format='%.8e')


if __name__ == '__main__':
    # Part 4 - Arbitrary Function
    arbitraryFunctionSegmentAnalysis()
