from context import fe621

import numpy as np
import pandas as pd


# Option characteristics
S = 100
K = 100
vol = 0.3
T = 1.
rf = 0.02

# FFT parameters
alpha = 1.1
N = 4096
k = np.log(K)
b = np.ceil(k)
lmbda = 2 * b / N
eta = 2 * np.pi / (N * lmbda)

# Values
x_j = np.zeros(N)
X_j = np.zeros(N)
k_u = np.array([-b + (lmbda * i) for i in range(0, N)])


# Phi
def phi(v, i):
    return np.exp(np.complex(0, np.complex(v, -(alpha + 1))) * (np.log(S) +
        (rf - 0.5 * vol) * T * i / N) - (0.5 * np.power(vol, 2) * 
        np.power(np.complex(v, -(alpha + 1)), 2)))

# Psi
def psi(v, i):
    return (np.exp(-rf * T * i / N) * phi(v, i)) / np.complex(np.power(alpha, 2) + alpha -
        np.power(v, 2), ((2 * alpha) + 1) * v)

# Computing adjusted values
for j in range(0, N):
    x_j[j] = np.exp(np.complex(0, b * eta * j)) * psi(j * eta, j) * eta

# Performing Fast Fourier Transform
X_j = np.fft.fft(x_j)

# Computing call option prices
C_k = np.exp(-alpha * k_u) / np.pi * X_j

# Isolating most accurate estimate
# for i in range(N):
#     if (np.abs(k_u[i] - np.log(K)) < 0.01):
#         print(i)
#         print(C_k[i].real)


# Isolting most accurate estimate
minarg = np.argmin(np.abs(k_u - k))
fft_price = C_k[minarg].real

# Computing traditional black-scholes price
bs_price = fe621.black_scholes.call(
    current=S,
    volatility=vol,
    ttm=T,
    strike=K,
    rf=rf
)

diff = np.abs(bs_price - fft_price)

# Building output dataframe, saving to CSV
pd.DataFrame({
    'Fast Fourier Transform Price': [fft_price],
    'Black Scholes Price': [bs_price],
    'Difference': [np.abs(diff)],
    '% Difference compared to BS': [str(round(diff * 100, 2)) + '%']
}, index=['Value']).T.round(decimals=7).to_csv(
    'Homework 3/bin/q2_price_comparison.csv')
