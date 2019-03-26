from context import fe621
from config import cfg

import pandas as pd

# Arbitrary option metadata
strike = 100
current = 100
ttm = 1
volatility = 0.25
rf = 0.06
dividend = 0.03
steps = 10

# Constructing arbitrary tree price
call_tree = fe621.tree_pricing.trinomial.AdditiveTree(current=current,
                                                      strike=strike,
                                                      ttm=ttm, rf=rf,
                                                      volatility=volatility,
                                                      opt_type='C',
                                                      opt_style='E',
                                                      dividend=dividend,
                                                      steps=steps)

prices = pd.DataFrame()

# Call Option prices
prices['European Call'] = [call_tree.getInstrumentValue()]
prices['American Call'] = [call_tree.computeOtherStylePrice(opt_style='A')]

put_tree = fe621.tree_pricing.trinomial.AdditiveTree(current=current,
                                                     strike=strike,
                                                     ttm=ttm, rf=rf,
                                                     volatility=volatility,
                                                     opt_type='P',
                                                     opt_style='E',
                                                     dividend=dividend,
                                                     steps=steps)

# Put option prices
prices['European Put'] = [put_tree.getInstrumentValue()]
prices['American Put'] = [put_tree.computeOtherStylePrice(opt_style='A')]

# Writing results to CSV
prices.T.to_csv('Homework 2/bin/trinomial_arbitrary_price.csv',
                index_label='Option Type', header=['Value'])
