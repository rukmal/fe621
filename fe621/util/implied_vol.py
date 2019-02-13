from . import option_metadata
from .. import black_scholes
from ..optimization import bisectionSolver

import numpy as np
import pandas as pd


def computeAvgImpliedVol(data: pd.DataFrame, name: str,
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
            imp_vol = bisectionSolver(f=optimFunc, a=0.0, b=2.0)

            # Appending to array
            imp_vols = np.append(imp_vols, imp_vol)

            print('option', column, 'time', index, 'imp_vol', imp_vol)

        # Computing mean implied volatility for option, adding to estimates
        estimates[column] = [np.mean(imp_vols)]

    # Cast to DataFrame, return
    return pd.DataFrame(estimates)
