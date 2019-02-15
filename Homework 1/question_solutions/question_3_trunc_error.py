from context import fe621

import numpy as np
import pandas as pd


def truncationErrorAnalysis():
    """Function to analyze the truncation error of the Trapezoidal and Simpson's
    quadature rules.
    """

    # Objective function
    def f(x: float) -> float:
        return np.where(x == 0.0, 1.0, np.sin(x) / x)
    
    # Setting values for N
    N = np.power(10, np.arange(3, 8))

    # Setting values for a
    a = np.power(10, np.arange(2, 7))

    trapezoidal_vals = np.ndarray((N.size, a.size))
    simpsons_vals = np.ndarray((N.size, a.size))

    # Building function approximation table, varying N and A
    for i in range(0, N.size):
        for j in range(0, a.size):
            # Trapezoidal rule approximation
            trapezoidal_vals[i, j] = fe621.numerical_integration \
                .trapezoidalRule(f=f, N=N[i], start=-a[j], stop=a[j])
            # Simpsons rule trunc approximation
            simpsons_vals[i, j] = fe621.numerical_integration \
                .simpsonsRule(f=f, N=N[i], start=-a[j], stop=a[j])
    
    # Computing the absolute difference from Pi (i.e. trunc error)
    # and casting to DataFrame
    trapezoidal_df = pd.DataFrame(np.abs(trapezoidal_vals - np.pi))
    simpsons_df = pd.DataFrame(np.abs(simpsons_vals - np.pi))

    # Setting row and column names
    trapezoidal_df.columns = ['N = ' + str(i) for i in N]
    trapezoidal_df.index = ['a = ' + str(i) for i in a]
    simpsons_df.columns = ['N = ' + str(i) for i in N]
    simpsons_df.index = ['a = ' + str(i) for i in a]

    # Saving to CSV
    trapezoidal_df.to_csv(
        'Homework 1/bin/numerical_integration/trapezoidal_trunc_error.csv',
        header=True, index=True, float_format='%.8e'
    )
    simpsons_df.to_csv(
        'Homework 1/bin/numerical_integration/simpsons_trunc_error.csv',
        header=True, index=True, float_format='%.8e'
    )


if __name__ == '__main__':
    # Part 2 - Truncation Error Analysis
    truncationErrorAnalysis()
