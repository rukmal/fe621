from context import fe621

import numpy as np
import pandas as pd


# Portfolio Metadata
port_init_val = 1e7  # Portfolio value
port_weights = np.array([.4, .3, .3])  # IBM, 10 yr Treasury, Yuan
initial_prices = np.array([80, 90000, 6.1])
asset_labels = ['IBM Equity', '10-Year T-Bill', 'CNY/USD ForEx']

# Portfolio initial stats
# Inverting CNY/USD rate as we're buying in USD
initial_prices_corrected = np.append(initial_prices[:-1], 1 / initial_prices[2])
# Flooring and casting to int as we can't buy fractional units
port_positions = np.floor((port_weights * port_init_val
    / initial_prices_corrected)).astype(int)

# Simulation data
sim_count = int(3e6)
dt = 0.001
t = 10 / 252
eval_count = int(np.ceil(t / dt))

# Processes
n_xt = lambda xt, w: xt + ((0.01 * xt * dt) + (0.3 * xt * np.sqrt(dt) * w))
n_yt = lambda yt, w, t: yt + (100 * (90000 + (1000 * t) - yt) * dt + (np.sqrt(yt)
    * np.sqrt(dt) * w))
n_zt = lambda zt, w: zt + ((5 * (6 - zt) * dt) + (0.01 * np.sqrt(zt)
    * np.sqrt(dt) * w))

# Function to compute portfolio value, given asset prices
def portfolioValue(asset_prices: np.array):
    # Making copy of asset prices (to not edit original array)
    asset_prices = np.copy(asset_prices)
    # Inverting last current price value (it is CNY/USD; we're buying in USD)
    asset_prices[2] = 1 / asset_prices[2]

    # Returning portfolio value (sum of elem-wise product across positions)
    return np.sum(np.multiply(asset_prices, port_positions))

# Simulation function
def sim_func(x: np.array) -> float:
    # Input: (3 x eval_count) matrix
    # Isolating initial prices (only need to maintain last observation)
    current_prices = np.copy(initial_prices)
    # Iterating over columns (i.e. in 3x1 chunks)
    for idx, w_vec in enumerate(x.T):
        current_prices = np.array([
            n_xt(current_prices[0], w_vec[0]),
            n_yt(current_prices[1], w_vec[1], dt * (idx + 1)),
            n_zt(current_prices[2], w_vec[2])
        ])
    
    return portfolioValue(current_prices)
        

# Running simulation
sim_data = fe621.monte_carlo.monteCarloSkeleton(
    sim_count=sim_count,
    eval_count=eval_count,
    sim_func=sim_func,
    sim_dimensionality=3
)


def exportInitialData():
    """Function to export initial portfolio data (answering question 2(a))
    """
    # Building output dictionary with necessary data
    # Specifically, positions, USD value, and CNY value
    output = dict()
    output['Positions'] = list(port_positions) + ['-']
    output['Position Value (USD)'] = np.append(np.multiply(
        initial_prices_corrected,
        port_positions
    ), portfolioValue(initial_prices))
    output['Position Value (CNY)'] = output['Position Value (USD)'] * initial_prices[2]

    # Building output dataframe, formatting and saving to CSV
    out_df = pd.DataFrame(output, index=[*asset_labels, 'Total'])
    out_df.to_csv('Homework 4/bin/q2_port_data.csv', float_format='%.0f')


def performRiskAnalytics():
    """Function to compute and export the portfolio VaR and CVaR (2(b) & 2(c))
    """
    
    # Output data dictionary
    output = dict()

    # VaR config
    N = 10
    alpha = 0.01

    # Computing simulation stats
    sim_stats = fe621.monte_carlo.monteCarloStats(mc_output=sim_data)

    # Computing value at risk (VaR) using the quantile method
    var = sim_stats['estimate'] - np.quantile(sim_data, alpha)
    var_daily = var / np.sqrt(N)
    output['VaR ($)'] = [var, var_daily]
    output['VaR (%)'] = np.array(output['VaR ($)']) / port_init_val * 100

    # Isolating portfolios that perform worse than the VaR risk threshold
    shortfall_ports = sim_data[sim_data <= np.quantile(sim_data, alpha)]
    # Computing conditional value at risk (cVaR) using the quantile method
    cvar = np.mean(sim_stats['estimate'] - shortfall_ports)
    cvar_daily = cvar / np.sqrt(N)
    output['CVaR ($)'] = [cvar, cvar_daily]
    output['CVaR (%)'] = np.array(output['CVaR ($)']) / port_init_val * 100

    # Building output dataframe, formatting and outputting to CSV
    out_df = pd.DataFrame(output, index=['10 Day', '1 Day']).T

    out_df.to_csv('Homework 4/bin/q2_risk_analytics.csv', float_format='%.4f')


if __name__ == '__main__':
    # Part (1)
    # exportInitialData()

    # Part (2)
    performRiskAnalytics()
