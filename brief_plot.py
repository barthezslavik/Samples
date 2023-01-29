import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/metrics/xgb_brief.csv')

# H,D,ROI H,ROI D,ROI A,Bets H,Bets D,Bets A,Profit H,Profit D,Profit A

# Plot
fig, ax = plt.subplots()
ax.plot(df['ROI H'], label='ROI H')
ax.plot(df['ROI D'], label='ROI D')
ax.plot(df['ROI A'], label='ROI A')
ax.set_xlabel('Bets')
ax.set_ylabel('ROI')
ax.legend()
plt.show()