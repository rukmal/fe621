from context import fe621

import numpy as np
import pandas as pd

# Option data
rf = 0.03
volatility = 0.3
current = 100
strike = 100
ttm = 5
days_in_year = 252

# Simulation data
sim_count = int(1e3)
eval_count = days_in_year * ttm

def analyticalCallPrice():
    analytical_price = fe621.black_scholes.asian.call(
        current=current,
        volatility=volatility,
        ttm=ttm,
        strike=strike,
        rf=rf,
        days_in_year=days_in_year
    )

    print(analytical_price)

def mcArithmeticCallPrice():
    # Need to figure out what they mean by "arithmetic" vs "geometric" here;
    # If we're using the different payoff comptuation formulas, we also need
    # to model the asset prices differently; one arithmetically the other geometrically.
    pass
        
def mcGeometricCallPrice():
    # Computing delta t
    dt = ttm / eval_count
    # Computing intitial value
    init_val = np.log(current)
    # Computing nudt
    nudt = (rf - (np.power(volatility, 2) / 2)) * dt

    # Lambda function to model Geometric Brownian Motion
    gbm = lambda x: nudt + (volatility * np.sqrt(dt) * x)

    # Initial log price
    init_log_price = np.log(current)

    # Defining simulation function
    def geom_sim_func(x: np.array) -> float:
        # Computing asset log-price and normal price path using GBM
        log_asset_price = np.cumsum(np.append(init_log_price, (gbm(x))))
        asset_price = np.append(current, current * np.exp(np.cumsum(gbm(x))))

        # Computing geometric average price of path
        avg_log_price = (1 / (eval_count + 1)) * np.sum(log_asset_price)
        # Computing geometric average price of the path
        avg_log_price = np.power(np.cumprod(asset_price)[-1], 1 / (eval_count + 1))

        # Computing arithmetic average price of path
        avg_price = (1 / (eval_count + 1)) * np.sum(asset_price)
        
        # Computing PV of terminal payoff
        geom_value = np.exp(-1 * rf * ttm) * np.maximum(0,
            np.exp(avg_log_price) - strike)
        arithmetic_value = np.exp(-1 * rf * ttm) * np.maximum(0,
            avg_price - strike)
        
        return np.array([geom_value, arithmetic_value])
    
    # Running simulation
    sim_data = fe621.monte_carlo.monteCarloSkeleton(
        sim_func=geom_sim_func,
        eval_count=eval_count,
        sim_count=sim_count
    )

    # Extracting price data for each type of computation
    geom_sim_data = sim_data[:, 0]
    arithmetic_sim_data = sim_data[:, 1]

    print('geometric sim data\n', fe621.monte_carlo.monteCarloStats(geom_sim_data))
    print('arithmetic sim data\n', fe621.monte_carlo.monteCarloStats(arithmetic_sim_data))

if __name__ == '__main__':
    # Part (a) - Analytical Formula Geometric Asian Call Option price
    # analyticalCallPrice()

    # Part (b) and (c) - MC Price computation
    mcGeometricCallPrice()
