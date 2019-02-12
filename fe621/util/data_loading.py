import os
import pandas as pd


def loadData(folder_path: str, date: str, start_time: str='9:30',
             end_time: str='16:00') -> pd.DataFrame:
    """Function to load complete price data for a given asset, from a given
    folder. This function loads all '*.csv' files from a given directory
    corresponding to instruments on a specific asset. Given a date, this
    returns a formatted DataFrame with 1 minute intervals from a start
    time to end time of the day, with each of the aligned prices in columns
    corresponding to the files they were sourced from. This function assumes
    dates and times are in the first column of the CSV file (headers 'Dates'),
    and that the prices are in the second column. The corresponding column in
    the final DataFrame is the name of the file it was read from. This function
    also forward and backward propagates prices from the last/first viable
    value if one is not available for a given minute.

    Arguments:
        folder_path {str} -- Path from which CSV files are to be ingested.
        date {str} -- Date the data was collected. This is encoded in the index
                      of the DataFrame (format: yyyy-mm-dd).

    Keyword Arguments:
        start_time {str} -- Start time (military time) (default: {'9:30'}).
        end_time {str} -- End time (military time) (default: {'16:00'}).

    Returns:
        pd.DataFrame -- Formatted DataFrame with aligned prices.
    """

    file_list = os.listdir(folder_path)  # Getting files

    # Removing non-CSV files from list (Assume one '.' in file name)
    file_list = [x for x in file_list if x.split('.')[1] == 'csv']

    # Defining full start and end time
    start = date + ' ' + start_time
    end = date + ' ' + end_time

    # Building DataFrame with correct index
    data_index = pd.DatetimeIndex(start=start, end=end, freq='1min')

    # Creating empty DataFrame with index
    data = pd.DataFrame(index=data_index)

    for file_name in file_list:
        # Isolating security name
        asset_name = file_name.split('.')[0]
        # Loading data to DataFrame, parsing dates
        candidate_data = pd.read_csv(os.path.join(folder_path, file_name),
                                     index_col='Dates', parse_dates=True)
        # Renaming index
        candidate_data.index.name = 'date'
        # Renaming data column
        candidate_data.columns = [asset_name]
        # Re-indexing with correct index, propagate values forward
        candidate_data.reindex(index=data_index)
        # Add to main data
        data = pd.concat([data, candidate_data], axis=1)

    # Forward-filling data (i.e. forward-propagate last viable value)
    data = data.fillna(method='ffill')
    # Backward-filling data
    data = data.fillna(method='backfill')

    # Restrict time to start_time and end_time
    data = data.between_time(start_time=start_time, end_time=end_time)

    return data
