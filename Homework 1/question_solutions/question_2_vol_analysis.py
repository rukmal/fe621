from context import fe621

import numpy as np
import pandas as pd


# Defining date
data1_date = '2019-02-06'

# Loading computed average daily implied volatilities
spy_imp_vol = pd.read_csv('Homework 1/bin/spy_data1_vol.csv',
                          index_col=False, header=0)
amzn_imp_vol = pd.read_csv('Homework 1/bin/amzn_data1_vol.csv',
                           index_col=False, header=0)

# Loading price information (for daily close)
spy_prices = pd.read_csv('Homework 1/data/DATA1/SPY/SPY.csv',
                         index_col=False, header=0)
amzn_prices = pd.read_csv('Homework 1/data/DATA1/AMZN/AMZN.csv',
                          index_col=False, header=0)

# Isolating daily close prices
spy_close = spy_prices.iloc[-1][1]
amzn_close = amzn_prices.iloc[-1][1]
print(amzn_close)
# Defining 'money-ness' ratio
# NOTE: This needs to be changed when more data is available
lower_bound_pct = 0.975
upper_bound_pct = 1.025

def analyzeVolAvg(data: pd.DataFrame, underlying_close: float) -> list:
    """Function to compute the average daily implied volatility of in-the-money
    and out-of-the-money options.
    
    Arguments:
        data {pd.DataFrame} -- Input data containing implied volatilities.
        underlying_close {float} -- Daily closing price of the underlying asset.
    
    Returns:
        list -- List containing [itm_avg_vol, otm_avg_vol].
    """

    # Computing upper and lower bounds for 'moneyness'
    lower_bound = underlying_close * lower_bound_pct
    upper_bound = underlying_close * upper_bound_pct

    # Isolating in-the-money and out-of-the-money options
    out_money_options = data[(data['strike'] < lower_bound) | \
        (data['strike'] > upper_bound)]
    in_money_options = data[(data['strike'] >= lower_bound) | \
        (data['strike'] <= upper_bound)]
    
    # Computing average daily implied volatility of in and out the money options
    otm_vol_avg = np.mean(out_money_options['implied_vol'])
    itm_vol_avg = np.mean(in_money_options['implied_vol'])

    return [itm_vol_avg, otm_vol_avg]

if __name__ == '__main__':
    # Computing average daily implied volatility for itm and otm options
    spy_avg = analyzeVolAvg(data=spy_imp_vol, underlying_close=spy_close)
    amzn_avg = analyzeVolAvg(data=amzn_imp_vol, underlying_close=amzn_close)

    # Building output DataFrame
    output = pd.DataFrame({
        'SPY': spy_avg,
        'AMZN': amzn_avg
    })
    # Renaming index
    output.index = ['In-the-money Options Average Daily Implied Vol',
                    'Out-of-the-money Options Average Daily Implied Vol']
    # Write to CSV
    output.to_csv('Homework 1/bin/itm_otm_vol_analysis.csv')
