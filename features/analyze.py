import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/good/ft_winner_h_test.csv")

win_df = pd.DataFrame()

# For 1 to 50 with step 0.1
for i in range(1, 1000, 1):
    n = i / 100
    sample_df = df[df.h_odd > n]
    # Profit
    win_df = win_df.append({'h_odd': n, 'profit': sample_df.profit.sum()}, ignore_index=True)

# Plot chart of distribution of h_odd and profit
plt.figure(figsize=(10, 6))
sns.lineplot(x='h_odd', y='profit', data=win_df)
plt.show()

# Plot chart of distribution of h_odd and profit_lr
# plt.figure(figsize=(10, 6))
# sns.distplot(df.h_odd, label='H Odd')
# sns.distplot(df.lr_profit, label='Profit')
# plt.legend()
# plt.show()