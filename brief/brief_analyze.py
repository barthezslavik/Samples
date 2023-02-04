import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/metrics/xgb_brief.csv')

# Drop all where A > 5
df = df[df['A'] <= 5]

# # Sort by profit H
# df = df.sort_values(by='Profit H', ascending=False)
# print(df.head(10))

# # Sort by profit D
# df = df.sort_values(by='Profit D', ascending=False)
# print(df.head(10))

# # Sort by profit A
# df = df.sort_values(by='Profit A', ascending=False)
# print(df.head(10))

# Sort by ROI H
df = df.sort_values(by='ROI H', ascending=False)
print(df.head(50))

# Sort by ROI D
df = df.sort_values(by='ROI D', ascending=False)
print(df.head(50))

# Sort by ROI A
df = df.sort_values(by='ROI A', ascending=False)
print(df.head(50))