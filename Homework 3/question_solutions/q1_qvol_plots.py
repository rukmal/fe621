from context import fe621

from mpl_toolkits.mplot3d import Axes3D
from typing import Callable
import matplotlib.pyplot as plt
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


# Defining transformed CDF
def probability(x: float, x0: float, t: float) -> float:
    return 1 / (sigma(x) * np.sqrt(2 * np.pi * t)) * (sigma(x0) / sigma(x)) \
        * np.exp((-0.5 / t * np.power(
        trapezoidalRule(s_integrand, x0, x, N), 2)) + Q() * t)

# Defining points
x = np.linspace(50, 60)
x0 = np.linspace(30, 80)

# Building meshgrid of points for evaluation
x, x0 = np.meshgrid(x, x0)

# Defining vector of t's for plotting
t_vec = np.arange(10, 41, 10)

# Part (a) Quadratic Vol Plots

for t in t_vec:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    transition_prob = probability(x=x, x0=x0, t=t)
    ax.plot_surface(x, x0, transition_prob)
    plt.tight_layout()
    plt.savefig(fname='Homework 3/bin/q1_quadvol_t_{0}.png'.format(t))
    plt.close()


# Part (b)

# Verifying that the finite difference approxiumations of the transition
# probability density satisfies the PDE

# Partial of density w.r.t. time
def partialT(x, x0, t, delT):
    return (probability(x, x0, t + delT) - probability(x, x0, t)) / delT

# Partial of density w.r.t. price
def partialX(x, x0, t, delX):
    return (probability(x + delX, x0, t) - probability(x, x0, t)) / delX


delX = delT = 1e-3
x = 50
x0 = 40
t = 20

# Computing difference
diff = np.abs(partialT(x, x0, t, delT) - (np.power(sigma(x), 2) * 0.5 *
    partialX(x, x0, t, delX)))

# Saving to CSV file
pd.DataFrame({
    'Absolute Difference between PDE and Finite Difference Approximation': \
        [diff]
}).to_csv('Homework 3/bin/q1_finite_diff_approx_verification.csv', index=False)
