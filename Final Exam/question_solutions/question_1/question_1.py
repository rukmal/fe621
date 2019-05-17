from context import fe621

import asian_option_mc

import itertools
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
sim_count = int(1e6)
eval_count = days_in_year * ttm

# Output file paths
out_files = {
    'analytical_call_price': 'Final Exam/bin/q1_analytical_call_price.csv',
    'mc_option_prices': 'Final Exam/bin/q1_mc_asian_option_prices.csv'
}

# Globally used functions and values

dt = ttm / eval_count  # Computing delta t
nudt = (rf - (np.power(volatility, 2) / 2)) * dt  # Computing nudt

# Lambda function to model Geometric Brownian Motion
gbm = lambda x: nudt + (volatility * np.sqrt(dt) * x)
# PV of terminal payoff of an Asian call option, given average price
asianCallPayoffPV = lambda avg_price: np.maximum(avg_price - strike, 0)


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
    output = pd.DataFrame({'Analytical Asian Call Price': [analytical_price]})
    output.to_csv(out_files['analytical_call_price'], index=False)


def mcCallPrices():
    """Part (b) and (c): Use Monte Carlo simulations to compute the Geometric
    and Arithmetic Asian Call option.
    """

    # MC simulation to price Geometric Asian Call Option
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

    # MC simulation to price Arithmetic Asian Call Option
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

    # Compute MC stats and CIs for both price estimates 
    geometric_results = fe621.monte_carlo.monteCarloStats(
        geom_sim_data,
        computeCIs=True
    )
    arithmetic_results = fe621.monte_carlo.monteCarloStats(
        arithmetic_sim_data,
        computeCIs=True
    )
    print(np.mean(arithmetic_sim_data))
    print(np.std(arithmetic_sim_data))
    # Converting CIs to strings (for output)
    ci_fields = ['ci_0.99', 'ci_0.95']
    for key, opt_type in itertools.product(ci_fields,
        [geometric_results, arithmetic_results]):
        opt_type[key] = ['{:.4f}'.format(i) for i in opt_type[key]]

    # Build output DataFrame
    output = pd.DataFrame({
        'Geometric Asian Call': geometric_results,
        'Arithmetic Asian Call': arithmetic_results
    })
    # Update index
    output.index = ['95% CI', '99% CI', 'Price Estimate', 'Standard Deviation',
                    'Standard Error']

    output.to_csv(out_files['mc_option_prices'], float_format='{:.4f}')

if __name__ == '__main__':
    # Part (a) - Analytical Formula Geometric Asian Call Option price
    # analyticalCallPrice()

    # Part (b) and (c) - MC Price computation
    mcCallPrices()
