from context import fe621

import pandas as pd


# Loading raw CSV of MC methods analysis
simple_mc_analysis = pd.read_csv(
    'Homework 4/bin/raw_mc_methods_analysis.csv')

# Creating formatted table
out_df = simple_mc_analysis.pivot_table(index=['method', 'opt_type'], values=['estimate', 'standard_error', 'time_elapsed'])

out_df.columns = ['Estimate', 'Std Error', 'Time Elapsed']

# Format, output to CSV
out_df.to_csv('Homework 4/bin/q1_mc_methods.csv', float_format='%.5f',
              index_label=['MC Method', 'Option Type'])
