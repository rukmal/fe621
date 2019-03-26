from context import fe621
from config import cfg

import pandas as pd


# Loading homework 2 data
hw2_data = pd.read_csv('Homework 2/hw2_data2.csv', index_col=0)

# Container to store prices
computed_prices = pd.DataFrame()

# Steps for tree construction
steps = 200

# Flags
counter = 0

# Iterating through each of the options, computing tree prices
for idx, row in hw2_data.iterrows():
    # Dictionary to store new row data
    price_data = dict()
    
    # Isolating name
    price_data['name'] = idx
    # Assigning black scholes price
    price_data['bs_price'] = row['bs_price']

    # Initializing tree
    tree = fe621.tree_pricing.trinomial.AdditiveTree(
        current=cfg.data2_price,
        strike=row['strike'],
        ttm=row['ttm'],
        rf=cfg.data2_rf,
        volatility=row['implied_vol'],
        opt_type=row['opt_type'],
        opt_style='E',
        steps=steps
    )

    # Setting implied volatility used
    price_data['implied_vol'] = row['implied_vol']
    price_data['opt_type'] = row['opt_type']
    price_data['strike'] = row['strike']
    # Assigning European and American price
    price_data['trinomial_E'] = tree.getInstrumentValue()
    price_data['trinomial_A'] = tree.computeOtherStylePrice(opt_style='A')

    # Appending new row to output DataFrame
    computed_prices = computed_prices.append(price_data, ignore_index=True)
    
    # Log
    counter += 1
    print('%f%% Complete - Trinomial tree price for EU option %s is %f' % \
          (counter / len(hw2_data.index) * 100, idx, price_data['trinomial_E']))

# Setting index to option name
computed_prices = computed_prices.set_index('name')

# Saving to CSV
computed_prices.to_csv('Homework 2/bin/trinomial_data2_prices.csv', index=True,
                       float_format='%.4f')
