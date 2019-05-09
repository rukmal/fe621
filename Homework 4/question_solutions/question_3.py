from context import fe621

import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm


# Asset basket data
init_prices = np.array([100, 101, 98])
mu_vec = np.array([0.03, 0.06, 0.02])
sigma_vec = np.array([0.05, 0.2, 0.15])
corr_mat = np.array([[1.0, 0.5, 0.2],
                     [0.5, 1.0, -0.4],
                     [0.2, -0.4, 1.0]])

# Performing Cholesky decomposition
L = cholesky(corr_mat, lower=True)

# Defining simulation parameters
dt = 1 / 365
ttm = 100 / 365
sim_count = 1000
eval_count = int(ttm / dt)
rf = 0.06

# Defining process function
st = lambda x, volatility, mu: (mu * dt) + (volatility * np.sqrt(dt) * x)

# NOTE: See the following link for how I figured out correlating the random
#       variables with the Cholesky decomposition of the correlation matrix.
# https://scipy-cookbook.readthedocs.io/items/CorrelatedRandomSamples.html

def partB():
    """Solution to 3(b)
    """

    # Defining simulation function
    def sim_func(x: np.array) -> np.array:
        # Correlating random variables
        x = np.dot(L, x)

        return np.array([init_prices[i] * np.exp(np.cumsum(
            st(x[i], sigma_vec[i], mu_vec[i])))
            for i in range(0, 3)])

    # Running simulation
    sim_results = fe621.monte_carlo.monteCarloSkeleton(
        sim_count=sim_count,
        eval_count=eval_count,
        sim_func=sim_func,
        sim_dimensionality=3
    )

    # Reshaping as per question specs
    # (rows: time step, col: simulation, z: asset)
    sim_results = np.swapaxes(sim_results, 0, 1)  # sims to columns
    sim_results = np.swapaxes(sim_results, 0, 2)  # assets to z, time to row

    # Importing required packages for plotting
    # Note: Doing this here so I can use the debugger in other sections
    #       without the python-framework macOS installation issue

    # Importing plotting libs
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Isolating data for each axis
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Simulation number
    sim = 1

    x_vals = sim_results[:, sim, 0]
    y_vals = sim_results[:, sim, 1]
    z_vals = sim_results[:, sim, 2]
    # Plotting surface
    ax.plot(x_vals, y_vals, z_vals)

    # Formatting plot
    ax.set_xlabel('Asset 1 Price ($)')
    # Setting y label
    ax.set_ylabel('Asset 2 Price ($)')
    # Setting z label
    ax.set_zlabel('Asset 3 Price ($)')

    # Setting plot dimensions to tight
    plt.tight_layout()

    # Saving to file
    plt.savefig(fname='Homework 4/bin/correlated_bm_path.png')

    # Closing plot
    plt.close()


def partC():
    """Solution to 3(c)
    """
    
    strike = 100
    a_weights = np.array([1 / 3] * 3)

    # Defining simulation function
    def sim_func(x: np.array) -> float:
        # Correlating random variables
        x = np.dot(L, x)

        # Computing terminal asset prices for each of the 3 correlated assets
        term_prices = np.array([init_prices[i] * np.exp(np.sum(
            st(x[i], sigma_vec[i], mu_vec[i])))
            for i in range(0, 3)])
        
        # Computing weighted basket price, and comparing to strike price
        term_price = np.sum(np.multiply(term_prices, a_weights))

        # Computing both put and call prices; returning
        call_price = np.exp(-1 * rf * ttm) * np.maximum(term_price - strike, 0)
        put_price = np.exp(-1 * rf * ttm) * np.maximum(strike - term_price, 0)

        return np.array([call_price, put_price])

    # Running simulation
    sim_results = fe621.monte_carlo.monteCarloSkeleton(
        sim_count=sim_count,
        eval_count=eval_count,
        sim_func=sim_func,
        sim_dimensionality=3
    )

    # Output dictionary
    output = dict()

    # Iterating over option types, computing MC stats for each
    for idx, opt_type in zip([0, 1], ['European Call', 'European Put']):
        output[opt_type] = fe621.monte_carlo.monteCarloStats(sim_results.T[idx])

    # Building output dataframe, formatting and saving to CSV
    out_df = pd.DataFrame(output)
    out_df.index = ['Estimate', 'Standard Deviation', 'Standard Error']
    out_df.to_csv('Homework 4/bin/q3_basket_option.csv')


def partD():
    """Solution to 3(d)
    """
    
    # Simulation constants
    strike = 100
    a_weights = np.array([1 / 3] * 3)
    barrier = 104

    # Defining simulation function
    def sim_func(x: np.array) -> float:
        # Correlating random variables
        x = np.dot(L, x)

        # Computing asset prices for each of the 3 correlated assets
        asset_prices = np.array([init_prices[i] * np.exp(np.cumsum(
            st(x[i], sigma_vec[i], mu_vec[i])))
            for i in range(0, 3)])

        # Condition 1 - testing asset 2 against barrier
        if np.any(np.greater(asset_prices[1], barrier)):
            # Option value is equal to EU call on asset 2
            return np.exp(-1 * rf * ttm) * np.maximum(0,
                asset_prices[1][-1] - strike)
        
        # Condition 2 - testing max of asset 2 against max of asset 3
        if (np.max(asset_prices[1]) > np.max(asset_prices[2])):
            # Option value is (asset 2 term price ^2 - K)+
            return np.exp(-1 * rf * ttm) * np.maximum(0,
                np.power(asset_prices[1][-1], 2) - strike)
        
        # Condition 3 - testing average price of asset 2 against asset 3
        if (np.mean(asset_prices[1]) > np.mean(asset_prices[2])):
            # Option value is (avg asset 2 price - K)+
            return np.exp(-1 * rf * ttm) * np.maximum(0,
                np.mean(asset_prices[1]) - strike)
        
        # Otherwise, option is vanilla call option on the basket (same as (c))
        term_price = np.sum(np.multiply(asset_prices[:, -1], a_weights))
        return np.exp(-1 * rf * ttm) * np.maximum(term_price - strike, 0)

    # Running simulation
    sim_results = fe621.monte_carlo.monteCarloSkeleton(
        sim_count=sim_count,
        eval_count=eval_count,
        sim_func=sim_func,
        sim_dimensionality=3
    )

    # Building output dataframe with stats, formatting and saving to CSV
    out_df = pd.Series(fe621.monte_carlo.monteCarloStats(sim_results))
    out_df.index = ['Estimate', 'Standard Deviation', 'Standard Error']
    out_df.to_csv('Homework 4/bin/q3_exotic_option_mc.csv')


if __name__ == '__main__':
    # 3(b)
    # partB()

    # 3(c)
    # partC()

    # 3(d)
    partD()
