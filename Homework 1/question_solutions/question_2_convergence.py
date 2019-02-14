from context import fe621

from datetime import datetime
import numpy as np
import pandas as pd


# Defining dates
data1_date = '2019-02-06'

# Loading DATA1
spy_data1 = fe621.util.loadData(folder_path='Homework 1/data/DATA1/SPY',
                                date=data1_date)
amzn_data1 = fe621.util.loadData(folder_path='Homework 1/data/DATA1/AMZN',
                                 date=data1_date)

# Loading Risk-free rate (effective federal funds rate)
rf = pd.read_csv('Homework 1/data/ffr.csv')

# Setting comparison tolerance level
tol = 1e-3

# Number of input options
input_count = len(spy_data1.columns) - 1

def compareConvergenceTime():
    """Function to compare the convergence times of the Newton and Bisection
    method solvers, on the SPY option chain.
    """

    # Newton's Method
    start = datetime.now().timestamp()
    spy_vol_newton = fe621.util.computeAvgImpliedVolNewton(
        data=spy_data1,
        name='SPY',
        rf=rf[data1_date][0],
        current_date=data1_date,
        tol=tol
    )
    end = datetime.now().timestamp()

    # Computing time  and number of options for Newton
    newton_time = end - start
    newton_count = spy_vol_newton.count(axis=0)[0]

    # Bisection Method
    start = datetime.now().timestamp()
    spy_vol_bisection = fe621.util.computeAvgImpliedVolNewton(
        data=spy_data1,
        name='SPY',
        rf=rf[data1_date][0],
        current_date=data1_date,
        tol=tol
    )
    end = datetime.now().timestamp()
    
    # Computing time and number of options for Bisection
    bisection_time = end - start
    bisection_count = spy_vol_bisection.count(axis=0)[0]

    # Building DataFrame, and saving to CSV
    convergence_table = pd.DataFrame({
        'Number of Input Options': [input_count, input_count],
        'Number of Output Options': [newton_count, bisection_count],
        'Number of Dropped Options': [input_count - newton_count,
                                      input_count - bisection_count],
        'Time Elapsed for Computation (s)': [newton_time, bisection_time],
        'Average Time per Option (s)': [newton_time / input_count,
                                        bisection_time / input_count]
    })
    convergence_table = convergence_table.T  # Transposing so cols are methods
    convergence_table.columns = ['Newton Method', 'Bisection Method']
    convergence_table.to_csv('Homework 1/bin/imp_vol_convergence.csv')

if __name__ == '__main__':
    compareConvergenceTime()
