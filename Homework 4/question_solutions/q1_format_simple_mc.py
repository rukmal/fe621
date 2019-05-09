from context import fe621

import pandas as pd


# Loading raw CSV of simple GBM analysis
simple_mc_analysis = pd.read_csv(
    'Homework 4/bin/raw_simple_mc_gbm_analysis.csv')

# Creating table of evaluation time
simple_mc_eval_time = simple_mc_analysis.pivot(
    index='sim_count',
    columns='eval_count',
    values='time_elapsed'
)

# Creating table of standard error
simple_mc_std_error = simple_mc_analysis.pivot(
    index='sim_count',
    columns='eval_count',
    values='standard_error'
)

# Renaming columns and index
simple_mc_eval_time.columns = [' '.join(['n = ', str(int(i))])
    for i in simple_mc_eval_time.columns]
simple_mc_eval_time.index = pd.Index(simple_mc_eval_time.index, dtype=int)
simple_mc_std_error.columns = [' '.join(['n = ', str(int(i))])
    for i in simple_mc_std_error.columns]
simple_mc_std_error.index = pd.Index(simple_mc_std_error.index, dtype=int)

# Formatting, output to CSV
simple_mc_eval_time.to_csv('Homework 4/bin/q1_simple_mc_time.csv',
    float_format='%.5f', index_label='Simulation Count')
simple_mc_std_error.to_csv('Homework 4/bin/q1_simple_mc_std_err.csv',
    float_format='%.5f', index_label='Simulation Count')
