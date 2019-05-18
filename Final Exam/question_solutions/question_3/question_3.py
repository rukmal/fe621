from context import fe621

import numpy as np
import pandas as pd


data_file = 'Final Exam/question_solutions/question_3/SPX.xls'

main_data = pd.read_excel(data_file, header=1)
rf = 0.0066 # 0.66 percent
current = 770.05
current_date = 39866


# Part (a) Implied Volatility Computation
def imp_vol_computation():
    """3(a) [i] imp vol computation
    """

    implied_vols = []

    for idx, row in main_data.iterrows():
        # Defining function to be optimized
        def optimFunc(x: float) -> float:
            return fe621.black_scholes.call(
                current=current,
                volatility=x,
                ttm=row['T'],
                strike=row['K'],
                rf=rf
            ) - row['Price']

        # Running optimization
        try:
            imp_vol = fe621.optimization.bisectionSolver(
                f=optimFunc,
                a=0,
                b=3
            )
        except:
            imp_vol = np.nan

        print('Computed implied volatility for idx {0} = {1}'.format(idx, imp_vol))

        implied_vols.append(imp_vol)


    # Appending implied volatilities to DataFrame, saving
    main_data['implied_vol'] = implied_vols

    # Saving to CSV
    main_data.to_csv('Final Exam/bin/q3_imp_vols.csv')


def imp_vol_scatterplot():
    """3(a) [ii] vol scatterplot
    """

    # Loading data from CSV
    vol_data = pd.read_csv('Final Exam/bin/q3_imp_vols.csv')

    # Importing matplotlib
    # Doing this here so I can use debug in other methods
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Getting figure and axes
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plotting scatterplot
    ax.scatter(vol_data['K'], vol_data['T'], vol_data['implied_vol'])

    # Plot axes format
    ax.set_xlabel('Strike ($)')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied Volatility')

    # Setting plot dimensions to tight
    plt.tight_layout()

    # Saving to file
    plt.savefig(fname='Final Exam/bin/q3_imp_vol_scatter.png')

    # Closing plot
    plt.close()


def imp_vol_surface_plot():
    """3(b) [ii] vol surface plot
    See: http://bit.ly/2WPOOID
    """

    # Loading data from CSV
    vol_data = pd.read_csv('Final Exam/bin/q3_imp_vols.csv')
    # Dropping rows with na data
    vol_data = vol_data.dropna(axis=0)

    # Importing matplotlib
    # Doing this here so I can use debug in other methods
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt

    # Scipy for cubic spline interpolation
    import scipy
    import scipy.interpolate

    # Building grid of points
    x_grid = np.linspace(min(vol_data['K']), max(vol_data['K']),
                         len(vol_data['K']))
    y_grid = np.linspace(min(vol_data['T']), max(vol_data['T']),
                         len(vol_data['T']))
    B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')

    # Interpolating with 2 dimensional cubic spline
    cubic_spline = scipy.interpolate.griddata(
        (vol_data['K'], vol_data['T']),
        (vol_data['implied_vol']),
        (B1, B2),
        method='cubic'
    )

    # Getting figure and Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plotting surface and points
    ax.plot_surface(B1, B2, cubic_spline, alpha=0.5, rstride=1, cstride=1)
    ax.scatter(vol_data['K'], vol_data['T'], vol_data['implied_vol'],
               marker='x', c='red', alpha=0.2)

    # Plot axes format
    ax.set_xlabel('Strike ($)')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied Volatility')

    # Setting plot dimensions to tight
    plt.tight_layout()

    # Initializing view azimuth and elevation (trial + error)
    ax.view_init(elev=35, azim=45)

    # Saving figure
    plt.savefig(fname='Final Exam/bin/q3_vol_surface.png')


def calculate_local_vol():
    # Loading data from CSV
    vol_data = pd.read_csv('Final Exam/bin/q3_imp_vols.csv')
    # Dropping rows with na data
    vol_data = vol_data.dropna(axis=0)

    # Building output dataframe
    local_vols = []

    # Iterating over rows
    for idx, row in vol_data.iterrows():
        dt = (1e-3 * row['T'])
        dK = (1e-3 * row['K'])
        d1, _ = fe621.black_scholes.util.computeD1D2(
            current=current,
            volatility=row['implied_vol'],
            ttm=row['T'],
            strike=row['K'],
            rf=rf
        )

        # Function to get implied vol w.r.t. T, K, and Price
        def impliedVol(T, K, price):    
            # Defining function to be optimized
            def optimFunc(x: float) -> float:
                return fe621.black_scholes.call(
                    current=current,
                    volatility=x,
                    ttm=T,
                    strike=K,
                    rf=rf
                ) - price

            # Running optimization
            return fe621.optimization.bisectionSolver(
                f=optimFunc,
                a=0,
                b=10
            )
        print('current ivol', row['implied_vol'])
        fd_ivol_wrtT = lambda t: impliedVol(t, row['K'], row['Price'])
        fd_ivol_wrtK = lambda k: impliedVol(row['T'], k, row['Price'])

        try:
            dSigdT = fe621.numerical_differentiation.firstDerivative(
                f=fd_ivol_wrtT,
                x=row['T'],
                h=dt
            )


            dSigdK = fe621.numerical_differentiation.firstDerivative(
                f=fd_ivol_wrtK,
                x=row['K'],
                h=dK
            )

            d2SigdK2 = fe621.numerical_differentiation.secondDerivative(
                f=fd_ivol_wrtK,
                x=row['K'],
                h=dK
            )
        except Exception:
            local_vols.append(np.nan)
            continue

        # Computing local volatility
        sig = row['implied_vol']
        K = row['K']
        T = row['T']

        # Computing local volatility
        local_vol = np.sqrt(((2 * sig * dSigdT * T) + np.power(sig, 2) +
            (2 * sig * rf * T * K * dSigdK)) / (np.power(
            (1 + (K * d1 * dSigdK * np.sqrt(T))), 2) + (np.power(K, 2) * T * sig
            * (d2SigdK2 - (d1 * np.power(dSigdK, 2) * np.sqrt(T))))))
        print(local_vol)

        # Appending to outout array
        local_vols.append(local_vol)


    # Adding local volatility to the dataframe
    vol_data['local_vol'] = local_vols

    # Saving as new csv
    vol_data.to_csv('Final Exam/bin/q3_local_vol.csv')

def local_vol_surface_plot():
    """3(d) [ii] vol surface
    See: http://bit.ly/2WPOOID
    """

    # Loading data from CSV
    vol_data = pd.read_csv('Final Exam/bin/q3_local_vols.csv')
    # Dropping rows with na data
    vol_data = vol_data.dropna(axis=0)

    # Importing matplotlib
    # Doing this here so I can use debug in other methods
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt

    # Scipy for cubic spline interpolation
    import scipy
    import scipy.interpolate

    # Building grid of points
    x_grid = np.linspace(min(vol_data['K']), max(vol_data['K']),
                         len(vol_data['K']))
    y_grid = np.linspace(min(vol_data['T']), max(vol_data['T']),
                         len(vol_data['T']))
    B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')

    # Interpolating with 2 dimensional cubic spline
    cubic_spline = scipy.interpolate.griddata(
        (vol_data['K'], vol_data['T']),
        (vol_data['local_vol']),
        (B1, B2),
        method='cubic'
    )

    # Getting figure and Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plotting surface and points
    ax.plot_surface(B1, B2, cubic_spline, alpha=0.5, rstride=1, cstride=1)
    ax.scatter(vol_data['K'], vol_data['T'], vol_data['local_vol'],
               marker='x', c='red', alpha=0.2)

    # Plot axes format
    ax.set_xlabel('Strike ($)')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Local Volatility')

    # Setting plot dimensions to tight
    plt.tight_layout()

    # Initializing view azimuth and elevation (trial + error)
    ax.view_init(elev=35, azim=45)

    # Saving figure
    plt.savefig(fname='Final Exam/bin/q3_local_vol_surface.png')

if __name__ == '__main__':
    # Part (a) [i] implied volatility computation
    # imp_vol_computation()

    # Part (a) [ii] implied volatility scatterplot
    # imp_vol_scatterplot()

    # Part (b) Implied volatility surface
    # imp_vol_surface_plot()

    # Part (d) Dupire's Local Vol
    calculate_local_vol()

    # Part (d) Cont. Dupire's Local Vol Plot
    local_vol_surface_plot()
