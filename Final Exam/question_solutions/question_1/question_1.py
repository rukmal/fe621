from context import fe621

import asian_option_mc

import itertools
import numpy as np
import pandas as pd
import time


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
    'mc_option_prices': 'Final Exam/bin/q1_mc_asian_option_prices.csv',
    'mc_adjusted_prices': 'Final Exam/bin/q1_mc_adjusted_prices.csv'
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
    start = time.time()
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
    geometric_time = time.time() - start

    # MC simulation to price Arithmetic Asian Call Option
    start = time.time()
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
    arithmetic_time = time.time() - start

    # Compute MC stats and CIs for both price estimates; timing
    geometric_results = fe621.monte_carlo.monteCarloStats(
        geom_sim_data,
        computeCIs=True
    )
    
    arithmetic_results = fe621.monte_carlo.monteCarloStats(
        arithmetic_sim_data,
        computeCIs=True
    )

    # Converting CIs to strings (for output)
    ci_fields = ['ci_0.99', 'ci_0.95']
    for key, opt_type in itertools.product(ci_fields,
        [geometric_results, arithmetic_results]):
        # opt_type[key] = ['{:.4f}'.format(i) for i in opt_type[key]]
        opt_type[key] = '; '.join(['{:.4f}'.format(i) for i in opt_type[key]])
        opt_type[key] = '(' + opt_type[key] + ')'

    # Adding time elapsed to the output DataFrame
    geometric_results['time_elapsed'] = geometric_time
    arithmetic_results['time_elapsed'] = arithmetic_time

    # Build output DataFrame
    output = pd.DataFrame({
        'Geometric Asian Call': geometric_results,
        'Arithmetic Asian Call': arithmetic_results
    })
    # Update index
    output.index = ['95% CI', '99% CI', 'Price Estimate', 'Standard Deviation',
                    'Standard Error', 'Time Elapsed']

    output.to_csv(out_files['mc_option_prices'], float_format='{:.4f}')


def regressionSlope(M: int):
    """Using M simulations and the exact same random valriables,
    compute the regression slope.
    """

    arithmetic_prices = []
    geometric_prices = []

    def sim_func(x: np.array):
        # Simulation function computing prices for both types of options with
        # the same random numbers
        arithmetic_prices.append(asian_option_mc.sim_func_arithmetic(
            x=x,
            gbm=gbm,
            computePayoffPV=asianCallPayoffPV,
            current=current
        ))
        
        geometric_prices.append(asian_option_mc.sim_func_geometric(
            x=x,
            gbm=gbm,
            computePayoffPV=asianCallPayoffPV,
            current=current
        ))


    # Running simulation with M iterations
    fe621.monte_carlo.monteCarloSkeleton(
        sim_count=M,
        eval_count=eval_count,
        sim_func=sim_func
    )

    # Casting to numpy arrays
    arithmetic_prices = np.array(arithmetic_prices)
    geometric_prices = np.array(geometric_prices)

    # Computing means
    a_mean = np.mean(arithmetic_prices)
    g_mean = np.mean(geometric_prices)

    # Computing b_star
    b_star = np.sum([((arithmetic_prices[i] - a_mean) * (geometric_prices[i] \
        - g_mean)) for i in range(0, len(arithmetic_prices))]) \
        / np.sum(np.power(arithmetic_prices - a_mean, 2))
    
    return b_star, arithmetic_prices, geometric_prices


def partDEF():
    """Solution to part (d) [single M value]
    """

    # Computing analytical option price
    analytical_price = fe621.black_scholes.asian.call(
        current=current,
        volatility=volatility,
        ttm=ttm,
        strike=strike,
        rf=rf,
        days_in_year=days_in_year
    )

    out_df = pd.DataFrame()

    for i in range(3, 7):
        out_row = dict()
        M = int(np.power(10, i))
        b_star, arithmetic_prices, geom_prices = regressionSlope(M=M)

        # Computing MC price esimates
        price_a = fe621.monte_carlo.monteCarloStats(
            arithmetic_prices)['estimate']
        price_g = fe621.monte_carlo.monteCarloStats(geom_prices)['estimate']

        # Computing error for the geometric Asian option
        geom_err = np.abs(analytical_price - price_g)
        
        # Computing modified arithmetic option price using the error and b_star
        price_a_adj = price_a - (b_star * geom_err)

        # Building output dictionary
        out_row['M'] = M
        out_row['sim_price_g'] = price_g
        out_row['sim_price_a'] = price_a
        out_row['b*'] = b_star
        out_row['E_g'] = geom_err
        out_row['price_a_adj'] = price_a_adj

        out_df = out_df.append(out_row, ignore_index=True)

        print(out_row)
    
    # Saving to CSV
    out_df.to_csv(out_files['mc_adjusted_prices'])


    


if __name__ == '__main__':
    # Part (a) - Analytical Formula Geometric Asian Call Option price
    # analyticalCallPrice()

    # Part (b) and (c) - MC Price computation
    # mcCallPrices()

    # Part (d), (e), (f)
    partDEF()
