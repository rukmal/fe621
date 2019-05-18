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

# Output DataFrame
def get_params():
    """Solution to 2(b)
    """

    theta_1 = []
    theta_2 = []

    for asset in ['bac', 'c', 'gs', 'jpm']:
        prices = pd.read_csv(raw_data_files[asset])['Adj Close']
        # Daily log return
        log_rets = np.diff(np.log(prices))
        theta_1.append(np.mean(log_rets))
        theta_2.append(np.std(log_rets))

    output_df = pd.DataFrame({
        'Theta_1': theta_1,
        'Theta_2': theta_2
    }, index=['BAC', 'C', 'GS', 'JPM'])

    output_df.round(decimals=4).to_csv('Final Exam/bin/q2_params.csv')

if __name__ == '__main__':
    # 2 (b)
    get_params()
