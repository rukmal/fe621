from context import fe621
from question_solutions.config import cfg

import pandas as pd

# Defining DATA2 date
data2_date = '2019-02-06'

# Loading required files
#======================

# Loading computed option prices for SPY DATA2
data2_computed_options = pd.read_csv('Homework 1/bin/data2/spy_prices.csv',
                                     index_col=0)
# Loading computed implied volatilities for SPY DATA1
data1_imp_vol = pd.read_csv('Homework 1/bin/spy_data1_vol.csv', index_col=0)

# Building new dataset for Homework 2
hw2_data = pd.DataFrame()

# Defining percent bounds for the considered options
lower_bound = 0.95
upper_bound = 1.05

# Iterating through rows of computed implied volatilities
for idx, imp_vol_data in data1_imp_vol.iterrows():
    # Dictionary to store row
    row_data = dict()

    # Assigning name
    row_data['name'] = idx
    # Assigning black scholes price
    row_data['bs_price'] = data2_computed_options.loc[idx]['computed_prices']
    # Assigning implied volatility
    row_data['implied_vol'] = imp_vol_data['implied_vol']
    # Assigning time to maturity and strike
    row_data['ttm'] = fe621.util.getTTM(name=idx, current_date=cfg.data2_date)
    row_data['strike'] = fe621.util.getStrikePrice(name=idx)
    row_data['opt_type'] = fe621.util.getOptionType(name=idx)

    # Adding row to new DataFrame if withing bounds
    if lower_bound <= row_data['strike'] / cfg.data2_price <= upper_bound:
        hw2_data = hw2_data.append(row_data, ignore_index=True)

# Setting index to option name
hw2_data = hw2_data.set_index('name')

# Saving homework 2 data to CSV
hw2_data.to_csv('Homework 2/hw2_data2.csv', index=True)
