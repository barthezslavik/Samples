import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/metrics/xgb_brief.csv')

# Sort by profit H
# df = df.sort_values(by='Profit H', ascending=False)

# Sort by profit D
# df = df.sort_values(by='Profit D', ascending=False)

# Sort by profit A
df = df.sort_values(by='Profit A', ascending=False)

print(df.head(20))