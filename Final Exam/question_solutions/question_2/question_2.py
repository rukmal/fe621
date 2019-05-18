from context import fe621

from scipy.linalg import cholesky
from scipy.stats import kurtosis, skew
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

# Output DataFrame
def get_params():
    """Solution to 2(b)
    """

    theta_1 = []
    theta_2 = []

    N = 255

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

    sim_count = 10
    eval_count = 255
    dt = 1 / 255

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


if __name__ == '__main__':
    # 2 (b)
    # get_params()

    # 2(c)
    # corr_matrix()

    # 2(d)
    eulerMilsteinSim()
