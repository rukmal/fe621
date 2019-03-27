from context import fe621

import pandas as pd


# Option parameters
current = 10
strike = 10
ttm = 0.3
volatility = 0.2
rf = 0.01
H = 11
steps = 200

q3_answers = pd.DataFrame()

# Part (a)
barrier_tree = fe621.tree_pricing.binomial.Barrier(
    current=current,
    strike=strike,
    ttm=ttm,
    rf=rf,
    volatility=volatility,
    barrier=H,
    barrier_type='O',
    opt_type='C',
    opt_style='E',
    steps=steps
)

q3_answers['Tree Up-and-Out EU Call'] = [barrier_tree.getInstrumentValue()]


# Part (b)
analytical_upandout = fe621.black_scholes.barrier.callUpAndOut(
    S=current,
    H=H,
    volatility=volatility,
    ttm=ttm,
    K=strike,
    rf=rf
)

q3_answers['Analytical Up-and-Out EU Call'] = [analytical_upandout]


# Part (c)
barrier_tree = fe621.tree_pricing.binomial.Barrier(
    current=current,
    strike=strike,
    ttm=ttm,
    rf=rf,
    volatility=volatility,
    barrier=H,
    barrier_type='I',
    opt_type='C',
    opt_style='E',
    steps=steps
)

q3_answers['Tree Up-and-In EU Call'] = [barrier_tree.getInstrumentValue()]

analytical_upandin = fe621.black_scholes.barrier.callUpAndIn(
    S=current,
    H=H,
    volatility=volatility,
    ttm=ttm,
    K=strike,
    rf=rf
)

q3_answers['Analytical Up-and-In EU Call'] = [analytical_upandin]


# Part (d)
barrier_tree = fe621.tree_pricing.binomial.Barrier(
    current=current,
    strike=strike,
    ttm=ttm,
    rf=rf,
    volatility=volatility,
    barrier=H,
    barrier_type='I',
    opt_type='C',
    opt_style='A',
    steps=steps
)

q3_answers['Tree Up-and-In A Call'] = [barrier_tree.getInstrumentValue()]


# Writing results to CSV
q3_answers.T.to_csv('Homework 2/bin/barrier_option_values.csv',
                index_label='Type', header=['Value'])
