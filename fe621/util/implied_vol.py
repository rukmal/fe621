from . import option_metadata
from .. import black_scholes
from ..optimization import bisectionSolver

import numpy as np
import pandas as pd


def computeAvgImpliedVolBisection(data: pd.DataFrame, name: str,
                         rf: float, current_date: str) -> pd.DataFrame:
    """Function to compute the average implied volatility of a series of Option
    contracts, in the form created in the `hw1_code.util.loadData` function.

    Arguments:
        data {pd.DataFrame} -- Price data in the form output by
                               `hw1_code.util.loadData`.
        name {str} -- Name of the underlying asset (ticker).
        rf {float} -- Risk-free rate.
        current_date {str} -- Current date (of data) in the form: YYYY-MM-DD.

    Returns:
        pd.DataFrame -- DataFrame with columns of option names, and
                        corresponding implied volatilities.
    """

    # Isolating asset prices
    prices = data[name]

    # Empty dictionary for implied volatility estimates
    estimates = dict()

    for column in data:
        # Skip prices column
        if column == name:
            continue

        # Computing ttm, strike, and type
        is_call = option_metadata.isCallOption(name=column)
        ttm = option_metadata.getTTM(name=column, current_date=current_date)
        strike = option_metadata.getStrikePrice(name=column)

        imp_vols = np.array([])

        for index, price in data[column].iteritems():
            # Defining function to be optimized
            def optimFunc(x) -> float:
                # Assigning price computation function based on type
                if is_call:
                    f = black_scholes.call
                else:
                    f = black_scholes.put

                # Returning computed option price less actual price
                return price - f(current=data[name][index], volatility=x,
                                 ttm=ttm, strike=strike, rf=rf)

            # Computing implied volatility for each price
            try:
                imp_vol = bisectionSolver(f=optimFunc, a=0.0, b=5.0)
            except Exception:
                print('WARNING: No implied vol solution found for {0} at {1}'
                      .format(column, index))
                continue

            # Appending to array
            imp_vols = np.append(imp_vols, imp_vol)

            print('option', column, 'time', index, 'imp_vol', imp_vol)

        # Computing mean implied volatility for option, adding to estimates
        estimates[column] = [np.mean(imp_vols)]

    # Cast to DataFrame, clean and return
    return cleanImpliedVol(candidate_df=pd.DataFrame(estimates))


def cleanImpliedVol(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Function to clean and format the DataFrame output by the implied
    volatility computation. Extracts option expiry, type, strike, and implied
    volatility and assigns them to individual columns in a new DataFrame.
    
    Arguments:
        candidate_df {pd.DataFrame} -- Output DataFrame from average implied
                                       volatility computation.
    
    Returns:
        pd.DataFrame -- Formatted DataFrame.
    """

    # Transposing and removing top (empty) row
    candidate_df = candidate_df.T[1:]

    # Creating empty dataframe to store formatted output
    clean_df = pd.DataFrame()

    # Iterating through transformed DataFrame, isolating option metadata
    for option_name, implied_vol in candidate_df.iterrows():
        # Extracting implied vol; it is the only element in the series
        implied_vol = implied_vol[0]

        # Extracting option metadata
        option_expiry = option_metadata.getExpiration(name=option_name) \
                        .strftime('%Y-%m-%d')
        
        # Extracting option type
        option_type = 'C' if option_metadata.isCallOption(name=option_name) else 'P'
        
        # Extracting option strike
        option_strike = option_metadata.getStrikePrice(name=option_name)

        # Appending to new DataFrame
        clean_df = clean_df.append(pd.Series([option_name, option_expiry,
                                              option_type, option_strike,
                                              implied_vol]), ignore_index=True)

    clean_df.columns = ['name', 'expiration', 'type', 'strike', 'implied_vol']

    return clean_df
