from . import option_metadata
from .. import black_scholes
from ..optimization import bisectionSolver, newtonSolver

import numpy as np
import pandas as pd


def computeAvgImpliedVolBisection(data: pd.DataFrame, name: str, rf: float,
                                current_date: str, tol: float) -> pd.DataFrame:
    """Function to compute the average implied volatility of a series of Option
    contracts, in the form created by the `util.loadData` function.

    Arguments:
        data {pd.DataFrame} -- Price data in the form output by `util.loadData`.
        name {str} -- Name of the underlying asset (ticker).
        rf {float} -- Risk-free rate.
        current_date {str} -- Current date (of data) in the form: YYYY-MM-DD.
        tol {float} -- Tolerance level of the estimate.

    Returns:
        pd.DataFrame -- DataFrame with columns of option metadata, and
                        corresponding implied volatilities.
    """

    # Empty dictionary for implied volatility estimates
    estimates = dict()

    for column in data:
        # Skip underlying prices column
        if column == name:
            continue

        # Computing ttm, strike, and type
        is_call = option_metadata.isCallOption(name=column)
        ttm = option_metadata.getTTM(name=column, current_date=current_date)
        strike = option_metadata.getStrikePrice(name=column)

        # Empty array to store computed implied volatilities
        imp_vols = np.array([])

        for index, price in data[column].iteritems():
            # Defining function to be optimized
            def optimFunc(x: float) -> float:
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
                imp_vol = bisectionSolver(f=optimFunc, a=0.0, b=5.0, tol=tol)
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


def computeAvgImpliedVolNewton(data: pd.DataFrame, name: str, rf: float,
                               current_date: str, tol: float) -> pd.DataFrame:
    """Function to compute the average implied volatility of a series of
    Option contracts, in the form created by the `util.loadData` function.
    
    Arguments:
        data {pd.DataFrame} -- Price data in the form output by `util.loadData`.
        name {str} -- Name of the underlying asset (ticker).
        rf {float} -- Risk-free rate.
        current_date {str} -- Current date (of data) in the form: YYYY-MM-DD.
        tol {float} -- Tolerance level of the estimate.
    
    Returns:
        pd.DataFrame -- DataFrame with columns of option metadata, and
                        corresponding implied volatilities.
    """

    # Empty dictionary for implied volatility estimates
    estimates = dict()

    for column in data:
        # Skip underlying prices column
        if column == name:
            continue
        
        # Computing ttm, strike, and type
        is_call = option_metadata.isCallOption(name=column)
        ttm = option_metadata.getTTM(name=column, current_date=current_date)
        strike = option_metadata.getStrikePrice(name=column)

        # Empty array to store computed implied volatilities
        imp_vols = np.array([])

        for index, price in data[column].iteritems():
            # Defining function to be optimized
            def optimFunc(x: float) -> float:
                # Assigning price computation function based on type
                if is_call:
                    f = black_scholes.call
                else:
                    f = black_scholes.put
                
                # Returning computed option price less actual price
                return f(current=data[name][index], volatility=x, ttm=ttm,
                         strike=strike, rf=rf) - price
            
            # Defining derivative of optimization function
            def optimFuncDerivative(x: float) -> float:
                return black_scholes.greeks.vega(current=data[name][index],
                                                 volatility=x,
                                                 ttm=ttm, strike=strike, rf=rf)
        
            # Computing implied volatility for each price
            try:
                imp_vol = newtonSolver(f=optimFunc, f_prime=optimFuncDerivative,
                                       guess=5, tol=tol)
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

    # Sorting by expiration, then by strike price
    clean_df.sort_values(by=['expiration', 'strike'], inplace=True)

    return clean_df
