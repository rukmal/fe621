from datetime import datetime
import os
import re


def renameOptionFiles(folder_path: str):
    """Function to rename files corresponding to option pricing data from the
    format output by the R Bloomberg data gathering utility to files with
    OOC (Options Clearing Commission) formatting. For RegEx, https://regexr.com
    eg: SPY 3-15-19 C270 Equity.csv -> SPY190315C00270000.csv
    
    Arguments:
        folder_path {str} -- Path to folder containing files to be renamed.
    """

    for filename in os.listdir(path=folder_path):
        # Isolate ticker
        ticker_search = '([^\s]+).*.csv'
        ticker = re.search(ticker_search, filename)[1]
        
        # Isolate option type and check if it is an option
        option_type_search = '.* ([CP])\d+.*.csv'
        option_type = re.search(option_type_search, filename)
        if option_type is None: continue  # Go to next if not option
        option_type = option_type[1]

        # Isolate strike price
        strike_price_search = '.* [CP](\d+).*.csv'
        strike_price_int = int(re.search(strike_price_search, filename)[1])
        strike_price = ('%05d' % strike_price_int) + '000'  # OOC format

        # Isolate date
        date_search = '.* ([\d-]*) .*.csv'
        date_text = re.search(date_search, filename)[1]
        date_object = datetime.strptime(date_text, '%m-%d-%y')
        date = date_object.strftime('%y%m%d')  # OOC format

        # Building OOC convention option name
        ooc_name = ticker + date + option_type + strike_price
        ooc_filename = ooc_name + '.csv'

        # Renaming file
        os.rename(src=folder_path + '/' + filename,
                  dst=folder_path + '/' + ooc_filename)
