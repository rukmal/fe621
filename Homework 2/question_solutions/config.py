from context import fe621

import pandas as pd


class cfg():
    """Configuration variables for Homework 2. Note that all 'data2' fields here
    correspond to SPY.
    """

    data2_date = '2019-02-06'
    data2_rf = pd.read_csv('Homework 1/data/ffr.csv')[data2_date][0]
    data2_price = pd.read_csv('Homework 1/data/DATA2/SPY/SPY.csv', index_col=0)\
                    .iloc[-1]['Close']
