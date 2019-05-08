from context import fe621

import numpy as np
import pandas as pd
import time


# Part (a)
def partA():
    """Function to answer Part (a) of question 1; analyzing the performance of
    simple GBM Black-Scholes model heuristic Monte Carlo option pricing.
    """

    # Initial parameters
    current = 100
    strike = 100
    volatility = .2
    rf = 0.06
    dividend = 0.03
    ttm = 1

    # Parameters for simulation analysis
    sim_counts = np.arange(start=1e6, stop=5e6 + 1, step=1e6, dtype=int)
    eval_counts = np.arange(start=300, stop=701, step=100, dtype=int)

    # Output dataframe
    output = pd.DataFrame()

    # Output monitoring stuff
    total_combos = len(sim_counts) * len(eval_counts)
    counter = 1

    # Iterating over possible simulation counts
    for sim_count in sim_counts:
        # Iterating over possible evaluation counts
        for eval_count in eval_counts:
            # Print update
            print('Starting simulation with eval count {0} and sim count {1}'.
                format(eval_count, sim_count))

            # Metadata dictionary
            meta = dict()

            # Storing simulation count and evaulation count
            meta['sim_count'] = sim_count
            meta['eval_count'] = eval_count

            # Starting timer
            start_time = time.time()
            
            # Running black scholes monte carlo simulation
            sim_output = fe621.monte_carlo.simple_gbm.blackScholes(
                current=current,
                volatility=volatility,
                ttm=ttm,
                strike=strike,
                rf=rf,
                dividend=dividend,
                sim_count=sim_count,
                eval_count=eval_count,
                opt_type='C'
            )

            # Recording time elapsed
            meta['time_elapsed'] = time.time() - start_time
            
            # Updating meta dictionary with values from simulation output
            meta.update(sim_output)

            # Adding to output dataframe
            output = output.append(meta, ignore_index=True)
    
            # Printing status update, update counter
            print('{0}% complete'.format(counter / total_combos * 100))
            print('Finished eval count {0} and sim count {1} in {2} minutes'.
                format(eval_count, sim_count, meta['time_elapsed'] / 60))
            counter += 1

    # Saving to CSV
    output.to_csv('Homework 4/bin/raw_simple_mc_gbm_analysis.csv', index=False)


if __name__ == '__main__':
    # Part A - raw data
    partA()
