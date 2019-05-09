from context import fe621

import numpy as np
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
eval_count = ttm / dt

# Defining process function
st = lambda x, volatility, mu: (mu * dt) + (volatility * np.sqrt(dt) * x)


def partB():
    """Solution to 3(b)
    """

    # Defining simulation function
    def sim_func(x: np.array) -> np.array:
        return np.array([init_prices[i] * np.exp(np.cumsum(
            st(x[i], sigma_vec[i], mu_vec[i])))
            for i in range(0, 3)])

    # Running simulation
    sim_results = fe621.monte_carlo.monteCarloSkeleton(
        sim_count=100,
        eval_count=10,
        sim_func=sim_func,
        sim_dimensionality=3
    )
    print(sim_results.shape)
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

if __name__ == '__main__':
    # 3(b)
    partB()
