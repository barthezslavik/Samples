import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    # read the data into a dataframe
    df = pd.read_csv('data/mean/nn.csv')

    # fit a polynomial regression of degree 2
    coefs = np.polyfit(df['Period'], df['SW'], 2)
    polynomial_regression = np.poly1d(coefs)

    # plot the data and the polynomial regression
    plt.scatter(df['Period'], df['SW'])
    plt.plot(df['Period'], polynomial_regression(df['Period']), 'r-')
    plt.xlabel('Period')
    plt.ylabel('SW')
    plt.title('Period vs SW')

    # Save plt as a png
    plt.savefig('data/mean/regression_nn_sw.png')
except:
    pass

try:
    # read the data into a dataframe
    df = pd.read_csv('data/mean/xgboost.csv')

    # fit a polynomial regression of degree 2
    coefs = np.polyfit(df['Period'], df['SW'], 2)
    polynomial_regression = np.poly1d(coefs)

    # plot the data and the polynomial regression
    plt.scatter(df['Period'], df['SL'])
    plt.plot(df['Period'], polynomial_regression(df['Period']), 'r-')
    plt.xlabel('Period')
    plt.ylabel('SL')
    plt.title('Period vs SL')

    # Save plt as a png
    plt.savefig('data/mean/regression_xgboost_sl.png')
    plt.close()

    # fit a polynomial regression of degree 2
    coefs = np.polyfit(df['Period'], df['D'], 2)
    polynomial_regression = np.poly1d(coefs)

    # plot the data and the polynomial regression
    plt.scatter(df['Period'], df['D'])
    plt.plot(df['Period'], polynomial_regression(df['Period']), 'r-')
    plt.xlabel('Period')
    plt.ylabel('D')
    plt.title('Period vs D')

    # Save plt as a png
    plt.savefig('data/mean/regression_xgboost_d.png')
except:
    pass