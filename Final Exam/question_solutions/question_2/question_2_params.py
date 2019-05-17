from context import fe621

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data_files = {
    'bac': 'Final Exam/question_solutions/question_2/raw_data/bac.csv',
    'c': 'Final Exam/question_solutions/question_2/raw_data/c.csv',
    'gs': 'Final Exam/question_solutions/question_2/raw_data/gs.csv',
    'jpm': 'Final Exam/question_solutions/question_2/raw_data/jpm.csv',
    'xlf': 'Final Exam/question_solutions/question_2/raw_data/xlf.csv'
}

dt = 1 / 255

data_bac = pd.read_csv(raw_data_files['bac'])['Adj Close']

print(data_bac.shape)

# BM Random Increments
bm = norm.rvs(size=data_bac.shape)
# Updating first position with 0 (for brownian motion)
bm = np.insert(bm, 0, 0)


# Building x axis (i.e. days * dt)
time = np.linspace(1, bm.shape[0], bm.shape[0])

plt.plot(time, bm)
plt.show()
