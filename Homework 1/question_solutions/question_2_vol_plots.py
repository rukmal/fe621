from context import fe621

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Loading implied volatility data from CSV files
spy_imp_vol = pd.read_csv('Homework 1/bin/spy_data1_vol.csv',
                          index_col=False, header=0)
amzn_imp_vol = pd.read_csv('Homework 1/bin/amzn_data1_vol.csv',
                           index_col=False, header=0)

# Defining date of DATA1
data1_date = '2019-02-06'


def plot2DVolSmile(data: pd.DataFrame, name: str, save_loc: str):
    """Function to plot a 2D Volatility Smile for a given option chain.
    
    Arguments:
        data {pd.DataFrame} -- Input data containing implied volatilities.
        name {str} -- Name of the underlying asset.
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

        ax = plt.gca()  # Get current axes

        # Setting y ticks and label
        ax.set_yticklabels(['{:,.1%}'.format(i) for i in ax.get_yticks()])
        ax.set_ylabel('Implied Volatility')
        # Setting x ticks and label
        ax.set_xticklabels(['$%i' % i for i in ax.get_xticks()])
        ax.set_xlabel('Strike Price')

        # Setting legend and setting plot dimensions to tight
        plt.legend()
        plt.tight_layout()
        
        # Saving to file
        full_option_type = 'Call' if (option_type == 'C') else 'Put'
        fname = '_'.join([name, full_option_type, '2DVolSmile.png'])
        plt.savefig(fname=(save_loc + '/' + fname))

        # Closing plot for next one
        plt.close()


def plot3DVolatilitySurface(data: pd.DataFrame, name: str, save_loc: str):
    """Fuction to plot a 3D Volatility Surface for a given option chain.
    
    Arguments:
        data {pd.DataFrame} -- Input data containing implied volatilities
        name {str} -- Name of the underlying asset.
        save_loc {str} -- Location (folder) to save the output image.
    """

    # Iterating through types of options for 2 separate put/call imp vol plots
    for option_type_group in data.groupby('type'):
        # Isolating current option type
        option_type = option_type_group[0]

        # Isolating plot data
        plot_data = option_type_group[1]

        # Creating new column with time to maturity information for each option
        ttm = plot_data.apply(lambda row: fe621.util.getTTM(
                                name=row.loc['name'],
                                current_date=data1_date),
                              axis=1)
        # Converting TTM to days
        ttm_days = ttm * 365

        # Isolating data for each axis
        x = np.array(ttm_days)
        y = np.array(plot_data['strike'])
        z = np.array(plot_data['implied_vol'])

        # Plotting surface
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(x, y, z, cmap='plasma')

        # Formatting plot
        #----------------
        
        # Setting x label
        ax.set_xlabel('TTM (Days)')
        # Setting y label
        ax.set_ylabel('Strike Price ($)')
        # Setting z label
        ax.set_zlabel('Implied Volatility')

        # Modifying z ticks to be percentages
        ax.set_zticklabels(['{:,.0%}'.format(i) for i in ax.get_zticks()])

        # Setting plot dimensions to tight
        plt.tight_layout()

        # Saving to file
        full_option_type = 'Call' if (option_type == 'C') else 'Put'
        fname = '_'.join([name, full_option_type, '3DVolSurface.png'])
        plt.savefig(fname=(save_loc + '/' + fname))

        # Closing plot for next one
        plt.close()

if __name__ == '__main__':
    # Plotting 2D Volatility Smile for AMZN and SPY option chains
    plot2DVolSmile(data=amzn_imp_vol, name='AMZN',
                   save_loc='Homework 1/bin/vol_smile/')
    plot2DVolSmile(data=spy_imp_vol, name='SPY',
                   save_loc='Homework 1/bin/vol_smile/')

    # Plotting 3D Volatility Surface for AMZN and SPY option chains
    plot3DVolatilitySurface(data=spy_imp_vol, name='SPY',
                            save_loc='Homework 1/bin/vol_surface/')
    plot3DVolatilitySurface(data=amzn_imp_vol, name='AMZN',
                            save_loc='Homework 1/bin/vol_surface/')
