import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/metrics/xgb_brief.csv')

# H,D,ROI H,ROI D,ROI A,Bets H,Bets D,Bets A,Profit H,Profit D,Profit A

# # Plot
# fig, ax = plt.subplots()
# ax.plot(df['ROI H'], label='ROI H')
# ax.plot(df['ROI D'], label='ROI D')
# ax.plot(df['ROI A'], label='ROI A')
# ax.set_xlabel('Bets')
# ax.set_ylabel('ROI')
# ax.legend()

# # Save plot
# plt.savefig('data/plots/brief_roi.png')

# fig, ax = plt.subplots()
# ax.plot(df['Profit H'], label='Profit H')
# ax.plot(df['Profit D'], label='Profit D')
# ax.plot(df['Profit A'], label='Profit A')
# ax.set_xlabel('Bets')
# ax.set_ylabel('Profit')
# ax.legend()

# # Save plot
# plt.savefig('data/plots/brief_profit.png')

# fig, ax = plt.subplots()
# ax.plot(df['Bets H'], label='Bets H')
# ax.plot(df['Bets D'], label='Bets D')
# ax.plot(df['Bets A'], label='Bets A')
# ax.set_xlabel('Bets')
# ax.set_ylabel('Bets')
# ax.legend()

# # Save plot
# plt.savefig('data/plots/brief_bets.png')

# # Correlation between ROI and A, D, H
# fig, ax = plt.subplots() # H -> 1.7
# ax.scatter(df['Profit H'], df['H'], label='Profit H')
# ax.scatter(df['Profit A'], df['H'], label='Profit A')
# ax.set_xlabel('Profit')
# ax.set_ylabel('Coefficient')
# ax.legend()

# Correlation between ROI and A, D, H
fig, ax = plt.subplots() # A -> 1.9
ax.scatter(df['Profit H'], df['A'], label='Profit H')
ax.scatter(df['Profit A'], df['A'], label='Profit A')
ax.set_xlabel('Profit')
ax.set_ylabel('Coefficient')
ax.legend()

# Save plot
plt.savefig('data/plots/brief_profit_coefficient.png')

# plt.show()