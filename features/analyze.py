import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/good/ft_winner_test.csv")
legends = []

df = df[df.prediction == 2]
for i in range(20, 35):
    n = i/5
    # Plot the profit
    df[f"profit_#{n}"] = df.profit
    df.loc[df.h_odd > n, f"profit_#{n}"] = 0
    legends.append(f"#{n}")
    df[f"profit_#{n}"].cumsum().plot()

# Legend
plt.legend(legends)
plt.show()