# Script to rename option files, from the R data download script format to
# OOC-compliant names.

from context import fe621

import os

option_file_paths = [os.getcwd() + i for i in ['/Homework 1/data/DATA1/AMZN',
                                               '/Homework 1/data/DATA1/SPY']]

for option_file_path in option_file_paths:
    fe621.util.renameOptionFiles(folder_path=option_file_path)
