import pandas as pd
import matplotlib.pyplot as plt

# read the data into a dataframe
df = pd.read_csv('data/mean/nn.csv')

# calculate the correlation matrix
corr = df.corr()

plt.scatter(df['Period'], df['SW'])
plt.xlabel('Period')
plt.ylabel('SW')
plt.title('Period vs SW')
# plt.show()

# Save plt as a png
plt.savefig('data/mean/nn_sw.png')

plt.close()

# if xgboost.csv exists, plot it
try:
    # read the data into a dataframe
    df = pd.read_csv('data/mean/xgboost.csv')

    # calculate the correlation matrix
    corr = df.corr()

    plt.scatter(df['Period'], df['D'])
    plt.xlabel('Period')
    plt.ylabel('D')
    plt.title('Period vs D')
    # plt.show()

    # Save plt as a png
    plt.savefig('data/mean/xgboost_d.png')
    plt.close()

    plt.scatter(df['Period'], df['SL'])
    plt.xlabel('Period')
    plt.ylabel('SL')
    plt.title('Period vs SL')
    # plt.show()

    # Save plt as a png
    plt.savefig('data/mean/xgboost_sl.png')
    plt.close()
except:
    pass