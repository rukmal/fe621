from context import fe621
from config import cfg

import pandas as pd


# Loading HW2 data
hw2_data = pd.read_csv('Homework 2/hw2_data2.csv', index_col=0)

# Container to store implied volatilities
implied_vol = pd.DataFrame()

# Iterating through options
for idx, row in hw2_data.iterrows():
    # Dictionary to store new row data
    imp_vol_data = dict()

    # Isolating name
    imp_vol_data['name'] = idx

    # Isolating type
    imp_vol_data['type'] = row['opt_type']

    imp_vol_data['strike'] = row['strike']

    # Setting steps
    steps = 50

    # Defining function to be optimized
    def optimFunc(x: float) -> float:
        # Building tree
        tree = fe621.tree_pricing.binomial.Trigeorgis(current=cfg.data2_price,
                                                      strike=row['strike'],
                                                      ttm=row['ttm'],
                                                      rf=cfg.data2_rf,
                                                      volatility=x,
                                                      opt_type=row['opt_type'],
                                                      opt_style='A',
                                                      steps=steps)
        
        return row['bs_price'] - tree.getInstrumentValue()

    try:
        imp_vol_data['binomial_vol'] = fe621.optimization.bisectionSolver(
            f=optimFunc, a=0.0, b=0.3, tol=0.001
        )
    except Exception:
        print('WARNING: No implied vol solution found for %s' % idx)
        continue

    # Appending to array
    implied_vol = implied_vol.append(imp_vol_data, ignore_index=True)

# Setting index to option name
implied_vol = implied_vol.set_index('name')

# Saving to CSV
implied_vol.to_csv('Homework 2/bin/binomial_implied_vol.csv',
                   float_format='%.4f')
