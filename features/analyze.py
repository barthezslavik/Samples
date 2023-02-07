import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/predictions/test_df.csv")

win_df = pd.DataFrame()

# For 1 to 50 with step 0.1
for i in range(1, 100, 1):
    n = i / 10
    sample_df = df[df.winning_odd < n]
    # Profit
    win_df = win_df.append({'winning_odd': n, 'profit': sample_df.lr_profit.sum()}, ignore_index=True)

# print(win_df)

# Plot chart of distribution of winning_odd and profit
plt.figure(figsize=(10, 6))
sns.lineplot(x='winning_odd', y='profit', data=win_df)
plt.show()

# Plot chart of distribution of winning_odd and profit_lr
# plt.figure(figsize=(10, 6))
# sns.distplot(df.winning_odd, label='Winning Odd')
# sns.distplot(df.lr_profit, label='Profit')
# plt.legend()
# plt.show()