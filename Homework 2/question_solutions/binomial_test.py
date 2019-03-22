from context import fe621

# Option stats (fake)

current = 90
strike = 100
volatility = 0.15
rf = 0.02
ttm = 0.25

# Testing price tree construction
test_tree = fe621.tree_pricing.binomial.Trigeorgis(current=current,
                                                   strike=strike,
                                                   volatility=volatility,
                                                   ttm=ttm,
                                                   rf=rf,
                                                   steps=2)

print(test_tree.getPriceTree())
