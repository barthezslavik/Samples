import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/metrics/xgb_brief_done.csv')

# H,D,ROI H,ROI D,ROI A,Bets H,Bets D,Bets A,Profit H,Profit D,Profit A

# Plot 3D: ROI H, ROI D, ROI A vs H, A
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['ROI H'], df['ROI D'], df['ROI A'], c='r', marker='o')
ax.set_xlabel('ROI H')
ax.set_ylabel('ROI D')
ax.set_zlabel('ROI A')
plt.show()
