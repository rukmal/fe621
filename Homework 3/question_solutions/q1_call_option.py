from context import fe621

from scipy.stats import norm
from typing import Callable
import numpy as np
import pandas as pd


# Setting parameters
alpha = 1e-4
beta = 1e-4
gamma = -1e-4
N = 100000

# Computing Q(t)
def Q(alpha: float=alpha, beta: float=beta, gamma: float=gamma) -> float:
    return ((alpha * gamma) / 2) - (np.power(beta, 2) / 8)

# Sigma(x)
def sigma(x: float, alpha: float=alpha, beta: float=beta, gamma: float=gamma) \
    -> float:
    return alpha * np.power(x, 2) + (beta * x) + gamma

# s(x) space-domain transformation
def s_integrand(x: float) -> float:
    return 1 / sigma(x)


# Note: Not using package function here as it is not compatible with
#       numpy meshgrid objects.
def trapezoidalRule(f: Callable, a: np.array, b: np.array, n: int) -> np.array:
    h = (b - a) / n
    integral = (0.5 * f(a)) + (0.5 * f(b))
    for i in range(1, n):
        integral += f(a + (i * h))
    integral *= h
    return integral

def qvolCall(T: float, K: float, x0: float):
    s = np.abs(trapezoidalRule(s_integrand, x0, K, N))
    return np.maximum(x0 - K, 0) + ((sigma(K) * sigma(x0)) / (2 * np.sqrt(-2 *
        Q())) * ((np.exp(s * np.sqrt(-1 * Q())) * norm.cdf((-1 * s / np.sqrt(2 *
        T)) - np.sqrt(-2 * Q() * T))) - (np.exp(-1 * s * np.sqrt(-1 * Q())) *
        norm.cdf((-1 * s / np.sqrt(2 * T)) + np.sqrt(-2 * Q() * T)))))


# Let the candidate option have the following characteristics:
S = 105
K = 100
vol = 0.03
T = 1.
rf = 0

bs_price = fe621.black_scholes.call(
    current=S,
    volatility=vol,
    ttm=T,
    strike=K,
    rf=rf
)

qvol_price = qvolCall(T=T, K=K, x0=S)


pd.DataFrame({
    'Black Scholes Price': [bs_price],
    'Quadratic Volatility Process Described Price': [qvol_price]
}).round(decimals=8).to_csv(
    'Homework 3/bin/q1_call_option_prices.csv', index=False)
