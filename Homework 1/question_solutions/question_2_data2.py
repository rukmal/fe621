from context import fe621

import pandas as pd


# Defining date
data2_date = '2019-02-06'

# Loading computed average daily implied volatilities
spy_options = pd.read_csv('Homework 1/bin/spy_data1_vol.csv',
                          index_col=False, header=0)
amzn_options = pd.read_csv('Homework 1/bin/amzn_data1_vol.csv',
                           index_col=False, header=0)

# Loading daily closing price information (for daily close)
spy_data2_close = pd.read_csv('Homework 1/data/DATA2/SPY/SPY.csv',
                              index_col=False, header=0).iloc[-1][1]
amzn_data2_close = pd.read_csv('Homework 1/data/DATA2/AMZN/AMZN.csv',
                               index_col=False, header=0).iloc[-1][1]

# Getting risk-free date (effective federal funds rate) for DATA2
rf = pd.read_csv('Homework 1/data/ffr.csv')[data2_date][0]


def computeData2Prices(data: pd.DataFrame, close: float) -> pd.DataFrame:
    """Function to compute the prices for a given set of options with implied
    volatilities, and a closing price.
    
    Arguments:
        data {pd.DataFrame} -- Input option data with implied volatility.
        close {float} -- Closing price of underlying asset.
    
    Returns:
        pd.DataFrame -- Formatted results DataFrame with DATA2 prices.
    """

    # Creating Series for results
    computed_prices = pd.Series()

    for idx, option_data in data.iterrows():
        # Isolating required arguments
        volatility = option_data['implied_vol']
        ttm = fe621.util.getTTM(name=option_data['name'],
                                current_date=data2_date)
        strike = fe621.util.getStrikePrice(name=option_data['name'])
        
        # Deciding price computation function based on type
        if option_data['type'] == 'C':
            computePrice = fe621.black_scholes.call
        else:
            computePrice = fe621.black_scholes.put

        # Computing price
        price = computePrice(current=close, volatility=volatility, ttm=ttm,
                             strike=strike, rf=rf)
        
        # Adding to output Series
        computed_prices.at[idx] = price
    
    # Copying `data` DataFrame for output
    results = data.copy(deep=True)

    # Dropping implied volatility column
    results.drop(labels=['implied_vol'], axis=1, inplace=True)

    # Adding computed prices
    results = results.assign(computed_prices=computed_prices)

    return results
    

if __name__ == '__main__':
    # Computing DATA2 prices for SPY
    spy_data2 = computeData2Prices(data=spy_options, close=spy_data2_close)

    # Saving to CSV
    spy_data2.to_csv('Homework 1/bin/data2/spy_prices.csv', index=False,
                     float_format='%.2f')

    # Computing DATA2 prices for AMZN
    amzn_data2 = computeData2Prices(data=amzn_options, close=amzn_data2_close)

    # Saving to CSV
    amzn_data2.to_csv('Homework 1/bin/data2/amzn_prices.csv', index=False,
                      float_format='%.2f')
