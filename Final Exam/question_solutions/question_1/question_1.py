from context import fe621

import asian_option_mc

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
sim_count = int(1e4)
eval_count = days_in_year * ttm

# Output file paths
out_files = {
    'analytical_call_price': 'Final Exam/bin/q1_analytical_call_price.csv',
    'mc_geom_call': 'Final Exam/bin/mc_geom_call.csv',
    'mc_arithmetic_call': 'Final Exam/bin/mc_arithmetic_call.csv'
}

def analyticalCallPrice():
    """Part (a): Use analytic formula to compute the option price.
    """

    # Computing the analytical price of the Asian Call option with the
    # fe621 package function
    analytical_price = fe621.black_scholes.asian.call(
        current=current,
        volatility=volatility,
        ttm=ttm,
        strike=strike,
        rf=rf,
        days_in_year=days_in_year
    )

    # Creating output dataframe, saving to CSV
    output = pd.Series({'Analytical Asian Call Price': analytical_price})
    output.to_csv(out_files['analytical_call_price'])

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

    # Function to compute the present value of the terminal value for an
    # Asian Call option
    asianCallPayoffPV = lambda avg_price: np.maximum(avg_price - strike, 0)

    geom_sim_data = fe621.monte_carlo.monteCarloSkeleton(
        sim_func=asian_option_mc.sim_func_geometric,
        eval_count=eval_count,
        sim_count=sim_count,
        sim_func_kwargs={
            'gbm': gbm,
            'computePayoffPV': asianCallPayoffPV,
            'current': current
        }
    )

    arithmetic_sim_data = fe621.monte_carlo.monteCarloSkeleton(
        sim_func=asian_option_mc.sim_func_arithmetic,
        eval_count=eval_count,
        sim_count=sim_count,
        sim_func_kwargs={
            'gbm': gbm,
            'computePayoffPV': asianCallPayoffPV,
            'current': current
        }
    )

    print('geometric sim data\n', fe621.monte_carlo.monteCarloStats(geom_sim_data))
    print('arithmetic sim data\n', fe621.monte_carlo.monteCarloStats(arithmetic_sim_data))

if __name__ == '__main__':
    # Part (a) - Analytical Formula Geometric Asian Call Option price
    analyticalCallPrice()

    # Part (b) and (c) - MC Price computation
    # mcGeometricCallPrice()
