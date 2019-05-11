from context import fe621

import numpy as np
import pandas as pd


# Defining objective function
def f(x: float) -> float:
    return np.exp(np.power(x, 2))

# Configuration variables for the quadrature rule approximations
start = 0
stop = 2
steps = 100

# Building DataFrame to store results
q1_res = pd.Series()

# Computing integral with the Trapezoidal rule
# Part (a)
q1_res.at['Trapezoidal Rule'] = fe621.numerical_integration.trapezoidalRule(
    f=f, N=steps, start=start, stop=stop
)

# Computing integral with Simpson's Rule
# Part (b)
q1_res.at['Simpson\'s Rule'] = fe621.numerical_integration.simpsonsRule(
    f=f, N=steps, start=start, stop=stop
)

# Updating Index label
q1_res.index.name = 'Quadrature Rule'

# Casting to DataFrame, saving to CSV
pd.DataFrame({'Estimated Area': q1_res}).to_csv(
    'Midterm Exam/bin/question_1.csv'
)
