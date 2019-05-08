from context import fe621

import numpy as np


# Portfolio Metadata
port_init_val = 1e7  # Portfolio value
port_weights = np.array([.4, .3, .3])  # IBM, 10 yr Treasury, Yuan
initial_prices = np.array([80, 90000, 6.1])

# Portfolio initial stats
# Inverting CNY/USD rate as we're buying in USD
initial_prices_corrected = np.append(initial_prices[:-1], 1 / initial_prices[2])
# Flooring and casting to int as we can't buy fractional units
port_positions = np.floor((port_weights * port_init_val
    / initial_prices_corrected)).astype(int)

# Simulation data
sim_count = int(1e2)
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

print(sim_data)

print(fe621.monte_carlo.monteCarloStats(sim_data))
