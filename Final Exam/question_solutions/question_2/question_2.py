from context import fe621

from scipy.stats import norm
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

    for asset in assets:
        prices = pd.read_csv(raw_data_files[asset])['Adj Close']
        # Daily log return
        log_rets = np.diff(np.log(prices))
        theta_1.append(np.mean(log_rets))
        theta_2.append(np.std(log_rets))

    output_df = pd.DataFrame({
        'Theta_1': theta_1,
        'Theta_2': theta_2
    }, index=asset_names)

    output_df.round(decimals=7).to_csv('Final Exam/bin/q2_params.csv')

def corr_matrix():
    log_rets = np.array([np.diff(np.log(pd.read_csv(
        raw_data_files[i])['Adj Close'])) for i in assets])
    
    corr_mat = np.corrcoef(log_rets)

    corr_mat_df = pd.DataFrame(corr_mat)

    corr_mat_df.index = asset_names
    corr_mat_df.columns = asset_names

    corr_mat_df.round(decimals=7).to_csv('Final Exam/bin/q2_corr_mat.csv')

if __name__ == '__main__':
    # 2 (b)
    get_params()

    # 2(c)
    corr_matrix()
