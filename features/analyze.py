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

# df = pd.read_csv("data/good/ft_winner_a_test.csv")

# win_df = pd.DataFrame()

# # For 1 to 50 with step 0.1
# for i in range(1, 1000, 1):
#     n = i / 100
#     sample_df = df[df.a_odd > n]
#     # Profit
#     win_df = win_df.append({'a_odd': n, 'profit': sample_df.profit.sum()}, ignore_index=True)

# # Plot chart of distribution of a_odd and profit
# plt.figure(figsize=(10, 6))
# sns.lineplot(x='a_odd', y='profit', data=win_df)
# plt.show()