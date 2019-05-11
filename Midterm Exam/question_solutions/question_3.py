from context import fe621

import numpy as np
import pandas as pd


# Option metadata
current = 23.35
strike = 22.5
ttm = (8 * 7) / 365  # In years, assuming 365 days per year
rf = 0.01

# Defining bid and ask prices from question
best_bid = 3.2
best_ask = 3.8

# Defining range of values 
a = 0
b = 2

# Computing possible implied volatilities using the bisection solver, and the
# Black-Scholes call option formula

implied_vol = pd.Series()

# Defining function to be optimized (bid)
def f_bid(x: float) -> float:
    return best_bid - fe621.black_scholes.call(
        current=current, volatility=x, ttm=ttm, strike=strike, rf=rf
    )

# Defining function to be optimized (ask)
def f_ask(x: float) -> float:
    return best_ask - fe621.black_scholes.call(
        current=current, volatility=x, ttm=ttm, strike=strike, rf=rf
    )


implied_vol.at['Bid'] = fe621.optimization.bisectionSolver(
    f=f_bid, a=a, b=b, tol=1e-4
)

implied_vol.at['Ask'] = fe621.optimization.bisectionSolver(
    f=f_ask, a=a, b=b, tol=1e-4
)

# Labeling Index
implied_vol.index.name = 'Initial Price'

# Saving implied volatility range to CSV
pd.DataFrame({'Computed Implied Volatility': implied_vol}).to_csv(
    'Midterm Exam/bin/question_3_imp_vol.csv'
)


# European Put Option Range - Part (a)

eu_put_range = pd.Series()

# Lower Bound
eu_put_range.at['Lower Bound'] = fe621.black_scholes.put(
    current=current, volatility=implied_vol['Bid'], ttm=ttm,
    strike=strike, rf=rf
)

# Upper Bound
eu_put_range.at['Upper Bound'] = fe621.black_scholes.put(
    current=current, volatility=implied_vol['Ask'], ttm=ttm,
    strike=strike, rf=rf
)

# Writing to CSV
pd.DataFrame({'Black-Scholes EU Put': eu_put_range}).to_csv(
    'Midterm Exam/bin/question_3_eu_put_prices.csv'
)


# American Call Option Range - Part (b)

a_call_range = pd.Series()

N = 200  # Steps for the tree

# Lower Bound
a_call_range.at['Lower Bound'] = fe621.tree_pricing.binomial.Trigeorgis(
    current=current, strike=strike, ttm=ttm, rf=rf,
    volatility=implied_vol['Bid'], opt_type='C', opt_style='A', steps=N
).getInstrumentValue()

# Upper Bound
a_call_range.at['Upper Bound'] = fe621.tree_pricing.binomial.Trigeorgis(
    current=current, strike=strike, ttm=ttm, rf=rf,
    volatility=implied_vol['Ask'], opt_type='C', opt_style='A', steps=N
).getInstrumentValue()

pd.DataFrame({'Trigeorgis Tree American Call': a_call_range}).to_csv(
    'Midterm Exam/bin/question_3_american_call_prices.csv'
)

