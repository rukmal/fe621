from context import fe621

from scipy.linalg import cholesky
from scipy.stats import kurtosis, skew
from sklearn import linear_model
import numpy as np
import pandas as pd

raw_data_files = {
    'bac': 'Final Exam/question_solutions/question_2/raw_data/bac.csv',
    'c': 'Final Exam/question_solutions/question_2/raw_data/c.csv',
    'gs': 'Final Exam/question_solutions/question_2/raw_data/gs.csv',
    'jpm': 'Final Exam/question_solutions/question_2/raw_data/jpm.csv',
    'xlf': 'Final Exam/question_solutions/question_2/raw_data/xlf.csv'
}

assets = ['bac', 'c', 'gs', 'jpm']
asset_names = ['BAC', 'C', 'GS', 'JPM']

N = 255

# Output DataFrame
def get_params():
    """Solution to 2(b)
    """

    theta_1 = []
    theta_2 = []

    for asset in assets:
        prices = pd.read_csv(raw_data_files[asset])['Adj Close']
        # Daily log return
        log_rets = np.diff(np.log(prices))
        theta_1.append(np.mean(log_rets) * N)
        theta_2.append(np.std(log_rets) * np.sqrt(N))

    output_df = pd.DataFrame({
        'Theta_1': theta_1,
        'Theta_2': theta_2
    }, index=asset_names)

    output_df.round(decimals=7).to_csv('Final Exam/bin/q2_params.csv')

def computeCorrMatrix() -> np.ndarray:
    """Helper to get correlation matrix of log returns of assets.
    """
    log_rets = np.array([np.diff(np.log(pd.read_csv(
        raw_data_files[i])['Adj Close'])) for i in assets])
    
    return np.corrcoef(log_rets)

def corr_matrix():
    """2(c) solution
    """
    corr_mat = computeCorrMatrix()

    corr_mat_df = pd.DataFrame(corr_mat)

    corr_mat_df.index = asset_names
    corr_mat_df.columns = asset_names

    corr_mat_df.round(decimals=7).to_csv('Final Exam/bin/q2_corr_mat.csv')


def eulerMilsteinSim():
    """2(d) solution
    """

    sim_count = 1000
    eval_count = N
    dt = 1 / N

    corr_mat = computeCorrMatrix()
    L = cholesky(corr_mat, lower=True)

    # Loading coefficients
    coefs = pd.read_csv('Final Exam/bin/q2_params.csv')

    # Loading initial prices
    init_prices = np.array([pd.read_csv(raw_data_files[i])['Adj Close'][0]
        for i in assets])

    # Extracting each of them, binding to numpy array
    theta_1 = np.array(coefs['Theta_1'])
    theta_2 = np.array(coefs['Theta_2'])

    # Defining simulation function
    def sim_func(x: np.array) -> np.array:
        # Entangling RVs
        x = np.dot(L, x)

        st = np.copy(init_prices)

        for time_step_rv in x.T:
            st = st + (np.multiply(st, theta_1) * dt) + (np.multiply(theta_2,
                np.multiply(time_step_rv, st)) * np.sqrt(dt))
    
        return st

    # Running simulation
    sim_results = fe621.monte_carlo.monteCarloSkeleton(
        sim_count=sim_count,
        eval_count=eval_count,
        sim_func=sim_func,
        sim_dimensionality=4
    )

    means = []
    sds = []
    skewness = []
    excess_kurt = []

    for asset_prices in sim_results.T:
        means.append(np.mean(asset_prices))
        sds.append(np.std(asset_prices))
        skewness.append(skew(asset_prices))
        excess_kurt.append(kurtosis(asset_prices))
    
    # Building output df
    out_df = pd.DataFrame({
        'Mean': means,
        'Standard Deviation': sds,
        'Skewness': skewness,
        'Excess Kurtosis': excess_kurt
    }, index=asset_names)

    # Saving to CSV
    out_df.round(decimals=7).to_csv('Final Exam/bin/q2_asset_stats.csv')

def etf_params():
    """ETF params solution to 2(e)
    """

    prices = pd.read_csv(raw_data_files['xlf'])['Adj Close']
    log_prices = np.diff(np.log(prices))

    out = dict()
    out['Theta_1'] = [np.mean(log_prices) * N]
    out['Theta_2'] = [np.std(log_prices) * np.sqrt(N)]

    # Saving to CSV
    pd.DataFrame(out, index=['XLF']).round(decimals=7).to_csv(
        'Final Exam/bin/q2_etf_params.csv')


def linear_regression() -> np.array:
    X = np.array([pd.read_csv(raw_data_files[i])['Adj Close'] for i in assets])
    y = np.array(pd.read_csv(raw_data_files['xlf'])['Adj Close'])[:, np.newaxis]

    lr = linear_model.LinearRegression(normalize=True)
    
    reg = lr.fit(X.T, y)

    return reg.coef_[0]


def exotic_option():
    sim_count = 1000
    eval_count = N
    dt = 1 / N

    corr_mat = computeCorrMatrix()
    L = cholesky(corr_mat, lower=True)

    # Loading coefficients
    coefs = pd.read_csv('Final Exam/bin/q2_params.csv')

    # Loading ETF coefficients
    etf_coefs = pd.read_csv('Final Exam/bin/q2_etf_params.csv')
    etf_theta1 = etf_coefs['Theta_1'][0]
    etf_theta2 = etf_coefs['Theta_2'][0]

    # Loading initial prices
    init_prices = np.array([pd.read_csv(raw_data_files[i])['Adj Close'][0]
        for i in assets])

    # Loading ETF initial price
    etf_init_price = pd.read_csv(raw_data_files['xlf'])['Adj Close'][0]

    # Extracting each of them, binding to numpy array
    theta_1 = np.array(coefs['Theta_1'])
    theta_2 = np.array(coefs['Theta_2'])

    weights = linear_regression()

    # Defining simulation function
    def sim_func(x: np.array) -> np.array:
        # Extracting random variables for ETF sim
        y = x[0]

        # Setting up RVs for asset basket
        x = x[1:]

        # Entangling RVs
        x = np.dot(L, x)

        st = np.copy(init_prices)

        xlf = etf_init_price

        for idx, time_step_rv in enumerate(x.T):
            st = st + (np.multiply(st, theta_1) * dt) + (np.multiply(theta_2,
                np.multiply(time_step_rv, st)) * np.sqrt(dt))
            xlf = xlf + (xlf * etf_theta1 * dt) + (etf_theta2 * y[idx] * xlf
                * np.sqrt(dt))

        basket_price = np.sum(np.multiply(weights, st))

        etf_exchange_opt = np.maximum(basket_price - xlf, 0)
        basket_exchange_opt = np.maximum(xlf - basket_price, 0)

        return [etf_exchange_opt, basket_exchange_opt]

    # Running simulation
    sim_results = fe621.monte_carlo.monteCarloSkeleton(
        sim_count=sim_count,
        eval_count=eval_count,
        sim_func=sim_func,
        sim_dimensionality=5
    )

    etf_exchange_opt = np.array(sim_results[:, 0])
    basket_exchange_opt = np.array(sim_results[:, 1])

    # Assuming risk free rate of 2%
    rf = 0.02
    etf_exchange_stats = fe621.monte_carlo.monteCarloStats(
        etf_exchange_opt * np.exp(rf * -1))
    basket_exchange_stats = fe621.monte_carlo.monteCarloStats(
        basket_exchange_opt * np.exp(rf * -1))

    # Output to CSV
    out = pd.DataFrame()
    out = out.append(etf_exchange_stats, ignore_index=True)
    out = out.append(basket_exchange_stats, ignore_index=True)

    out.index = ['ETF Exchnage Option', 'Basket Exchange Option']

    out.to_csv('Final Exam/bin/q2_exotic_option.csv')


if __name__ == '__main__':
    # 2 (2)
    # get_params()

    # 2(3)
    # corr_matrix()

    # 2(4)
    # eulerMilsteinSim()

    # 2(5)
    # etf_params()

    # 2(6)
    # linear_regression()

    # 2(7)
    exotic_option()
