from context import fe621

import matplotlib.pyplot as plt
import pandas as pd


# Loading implied volatility data from CSV files
spy_imp_vol = pd.read_csv('Homework 1/bin/spy_data1_vol.csv',
                          index_col=False, header=0)
amzn_imp_vol = pd.read_csv('Homework 1/bin/amzn_data1_vol.csv',
                           index_col=False, header=0)


def plot2DVolSmile(data: pd.DataFrame, name: str, save_loc: str):
    """Function to plot a 2D Volatility Smile for a given option chain.
    
    Arguments:
        data {pd.DataFrame} -- Input data containing implied volatilities.
        name {str} -- Name of the asset.
        save_loc {str} -- Location (folder) to save the output image.
    """

    # Iterating through types of options for 2 separate put/call imp vol plots
    for option_type_group in data.groupby('type'):
        # Isolating current option type
        option_type = option_type_group[0]
        
        # Iterating through expiration dates for individual lines for each
        for exp_date_group in option_type_group[1].groupby('expiration'):
            # Isolating current expiration date
            exp_date = exp_date_group[0]

            # Sorting data to be ascending on 'strike'
            plt_data = exp_date_group[1].sort_values(by='strike')

            # Plotting strike vs implied vol
            plt.plot(plt_data['strike'], plt_data['implied_vol'],
                     label=('Maturity on ' + exp_date))
        
        # Formatting plot
        #================

        ax = plt.gca()  # Get  current axes

        # Setting y ticks
        ax.set_yticklabels(['{:,.2%}'.format(i) for i in ax.get_yticks()])
        # Setting x ticks
        ax.set_xticklabels(['$%i' % i for i in ax.get_xticks()])

        # Setting legend and title
        plt.legend()
        full_option_type = 'Call' if (option_type == 'C') else 'Put'
        plt.title(' '.join([name, full_option_type, 'Implied Volatility']))

        fname = '_'.join([name, full_option_type, '2DVolSmile.png'])
        plt.savefig(fname=(save_loc + '/' + fname))
        plt.close()


if __name__ == '__main__':
    # Plotting 2D Volatility smile for AMZN and SPY option chains
    plot2DVolSmile(data=amzn_imp_vol, name='AMZN',
                   save_loc='Homework 1/bin/vol_smile/')
    plot2DVolSmile(data=spy_imp_vol, name='SPY',
                   save_loc='Homework 1/bin/vol_smile/')
