from context import fe621
from config import cfg

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Loading homework 2 data
hw2_data = pd.read_csv('Homework 2/hw2_data2.csv', index_col=0)

# Container to store prices
computed_prices = pd.DataFrame()

# Steps for tree construction
steps = [10, 20, 30, 40, 50,100, 150, 200, 250, 300, 350, 400]

# Candidate Put option metadata
option_name = 'SPY190315P00265000'
strike = 265.0
ttm = fe621.util.getTTM(name=option_name, current_date=cfg.data2_date)
implied_vol = hw2_data.loc[option_name]['implied_vol']

option_bs_price = fe621.black_scholes.put(current=cfg.data2_price,
                                          volatility=implied_vol,
                                          ttm=ttm,
                                          strike=strike,
                                          rf=cfg.data2_rf)

def computeAbsError() -> np.array:
    tree_prices = list()

    # Iterate through steps
    for step in steps:
        # Constructing tree
        candidate_tree = fe621.tree_pricing.binomial.Trigeorgis(
            current=cfg.data2_price, strike=strike, ttm=ttm, rf=cfg.data2_rf,
            volatility=implied_vol, opt_type='P', opt_style='E', steps=step
        )

        # Adding price to array for analysis
        tree_prices += [candidate_tree.getInstrumentValue()]

    # Casting to numpy array
    tree_prices = np.array(tree_prices)

    # Computing absolute error
    abs_error = np.abs(tree_prices - option_bs_price)

    # Building output dataframe
    abs_error_df = pd.DataFrame({'Steps': steps, 'Abs Error': abs_error})\

    # Saving to CSV
    abs_error_df.to_csv(
        'Homework 2/bin/binomial_tree_abs_error.csv', index=False)

    return abs_error_df


def plotAbsError(steps: np.array, abs_error: np.array):
    # Equation label
    eq_label = r'$\epsilon_N=\left|P^{BSM}(\cdot)-P^{BTree}_N(\cdot)\right|$'

    # Plotting points
    plt.plot(steps, abs_error, 'x--', label=eq_label, markeredgecolor='r')

    ax = plt.gca()  # Getting current axes

    # Setting x and y labels
    ax.set_xlabel(r'Number of Steps, $N$')
    ax.set_ylabel(r'Absolute Error, $\epsilon_N$')

    # Setting layout to tight
    plt.tight_layout()

    # Adding plot legend
    plt.legend(loc='upper right')

    # Save to file
    plt.savefig(fname='Homework 2/bin/binomial_abs_error_plot.png')

    # Clsoe plot
    plt.close()

if __name__ == '__main__':
    # Compute/load absolute error (uncomment relevant line)
    abs_error = computeAbsError()
    # abs_error = pd.read_csv('Homework 2/bin/binomial_tree_abs_error.csv',
    #                         index_col=False, header=0)

    # Plot graph of absolute error
    plotAbsError(steps=abs_error['Steps'].values,
                 abs_error=abs_error['Abs Error'].values)
