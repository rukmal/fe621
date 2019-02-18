from context import fe621

import pandas as pd


# Defining date
data1_date = '2019-02-06'

# Loading computed average daily implied volatilities
spy_options = pd.read_csv('Homework 1/bin/spy_data1_vol.csv',
                          index_col=False, header=0)
amzn_options = pd.read_csv('Homework 1/bin/amzn_data1_vol.csv',
                           index_col=False, header=0)

# Isolating call options
spy_call_options = spy_options[spy_options['type'] == 'C']
amzn_call_options = amzn_options[amzn_options['type'] == 'C']

# Loading price information (for daily close)
spy_prices = pd.read_csv('Homework 1/data/DATA1/SPY/SPY.csv',
                         index_col=False, header=0)
amzn_prices = pd.read_csv('Homework 1/data/DATA1/AMZN/AMZN.csv',
                          index_col=False, header=0)

# Isolating daily close prices
spy_close = spy_prices.iloc[-1][1]
amzn_close = amzn_prices.iloc[-1][1]

# Loading Risk-free rate (effective federal funds rate) for DATA1
rf = pd.read_csv('Homework 1/data/ffr.csv')[data1_date][0]

# Step size for computation
h = 1e-5

def computeAnalyticalAndEstimatedGreeks(data: pd.DataFrame, close: float) \
    -> pd.DataFrame:
    """Function to compute the Greeks for a given set of call options. It does
    this both using the analytical formulas and by numerical approximation. It
    uses the central finite difference method. It computes the Delta
    (first derivative w.r.t. underlying price), Gamma (second derivative w.r.t.
    underlying price), and the Vega (first derivative w.r.t. volatility).
    
    Arguments:
        data {pd.DataFrame} -- Option DataFrame with implied volatilities.
        close {float} -- Closing price of the underlying asset.

    Returns:
        pd.DataFrame -- Formatted DataFrame with computed results.
    """

    # Creating DataFrame for results
    results = pd.DataFrame()

    for _, option_data in data.iterrows():
        # Isolating required arguments
        volatility = option_data['implied_vol']
        ttm = fe621.util.getTTM(name=option_data['name'],
                                current_date=data1_date)
        strike = fe621.util.getStrikePrice(name=option_data['name'])

        # Computing analytical (prefix: a_*) and estimated (prefix: e_*) greeks

        # Delta (first derivative w.r.t. underlying price, S)
        a_delta = fe621.black_scholes.greeks.callDelta(current=close,
                                                       volatility=volatility,
                                                       ttm=ttm,
                                                       strike=strike,
                                                       rf=rf)
        e_delta = fe621.numerical_differentiation.firstDerivative(
            f=lambda x: fe621.black_scholes.call(
                x, volatility, ttm, strike, rf),
            x=close,
            h=h
        )

        # Gamma (second derivative w.r.t. underlying price, S)
        a_gamma = fe621.black_scholes.greeks.callGamma(current=close,
                                                       volatility=volatility,
                                                       ttm=ttm,
                                                       strike=strike,
                                                       rf=rf)
        e_gamma = fe621.numerical_differentiation.secondDerivative(
            f=lambda x: fe621.black_scholes.call(
                x, volatility, ttm, strike, rf),
            x=close,
            h=h
        )

        # Vega (first derivative w.r.t. volatility, $\sigma$)
        a_vega = fe621.black_scholes.greeks.vega(current=close,
                                                 volatility=volatility,
                                                 ttm=ttm,
                                                 strike=strike,
                                                 rf=rf)
        e_vega = fe621.numerical_differentiation.firstDerivative(
            f=lambda x: fe621.black_scholes.greeks.vega(
                close, x, ttm, strike, rf),
            x=volatility,
            h=h
        )

        # Adding to output DataFrame
        results = results.append(pd.Series([option_data['name'],
                                            a_delta, a_gamma,
                                            a_vega, e_delta,
                                            e_gamma, e_vega]),
                                 ignore_index=True)

    # Setting column names
    results.columns = ['name',
                       'delta_analytical', 'gamma_analytical',
                       'vega_analytical', 'delta_estimated',
                       'gamma_estimated', 'vega_estimated']

    return results
        

if __name__ == '__main__':
    # Computing Greeks for SPY
    spy_greeks = computeAnalyticalAndEstimatedGreeks(data=spy_call_options,
                                                     close=spy_close)
    # Saving to CSV
    spy_greeks.to_csv('Homework 1/bin/greeks/spy_greeks.csv', index=False,
                      float_format='%.7f')

    # Computing Greeks for AMZN
    amzn_greeks = computeAnalyticalAndEstimatedGreeks(data=amzn_call_options,
                                                      close=amzn_close)
    # Saving to CSV
    amzn_greeks.to_csv('Homework 1/bin/greeks/amzn_greeks.csv', index=False,
                       float_format='%.7f')
