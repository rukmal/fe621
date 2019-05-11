from context import fe621

import pandas as pd


# Configuring option data for tree construction
current = 100
strike = 120
ttm = 8 / 12  # Fraction of a year (assuming 365 days)
rf = 0
volatility = .3
N = 200


# Constructing Trinomial tree to price American Put option
tree = fe621.tree_pricing.trinomial.AdditiveTree(
    current=current, strike=strike, ttm=ttm, rf=rf, volatility=volatility,
    opt_type='P', opt_style='A', steps=N
)

# Part (b)
pd.DataFrame({'American Put Estimated Price': [tree.getInstrumentValue()]})\
    .to_csv('Midterm Exam/bin/question_2_price.csv', index=False)


# Part (c)

# Defining objective function for second derivative (Gamma) computation
def f(x: float) -> float:
    # Building trinomial tree for the given strike price, `x`, keeping
    # all other parameters the same
    tree = fe621.tree_pricing.trinomial.AdditiveTree(
        current=x, strike=strike, ttm=ttm, rf=rf, volatility=volatility,
        opt_type='P', opt_style='A', steps=N
    )
    return tree.getInstrumentValue()

# Computing estimate for Gamma
gamma = fe621.numerical_differentiation.secondDerivative(f=f, x=current, h=3)

# Writing to CSV
pd.DataFrame({'American Put Gamma Approximation': [gamma]}).to_csv(
    'Midterm Exam/bin/question_2_gamma.csv', index=False
)

