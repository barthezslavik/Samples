import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Read in dataset
dataset = pd.read_csv(f"data/test.csv", header=0)

# Transorm Date to_datetime
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d-%m-%Y')

# Add column year
dataset['year'] = dataset['Date'].dt.year

# Drop rows where year not in [2018, 2019, 2020, 2021, 2022]
# dataset = dataset[dataset['year'].isin([2018, 2019, 2020, 2021, 2022])]

# Sort by Date
dataset = dataset.sort_values(by=['Date'])

print(f'Number of rows: {len(dataset)}')

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Get only data
X_test = data.drop(['Date','Div', 'year', 'HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)
y_test = data['Y']

# Load model from file 'model/nn_model.sav'
nn_model = pickle.load(open('data/models/nn_model_global.sav', 'rb'))
y_pred = nn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Load model from file 'model/xgb_model.sav'
xgb_model = pickle.load(open('data/models/xgb_model_global.sav', 'rb'))
y_pred2 = xgb_model.predict(X_test)
y_pred2 = np.round(y_pred2).astype(int)

# Merge y_pred and y_pred2, if y_pred == 2, use y_pred2, else use y_pred
y_pred3 = np.where(y_pred == 2, y_pred2, y_pred)
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred3})

# Drop all rows except y_pred == 2
df = df[df['y_pred'] == 2]

# Add column called correct and set to 1 if y_test == y_pred
df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

# Merge with original dataset
df2 = pd.merge(df, dataset, left_index=True, right_index=True)

print(f'Number of rows: {len(df2)}')

# Drop all columns except y_test, y_pred, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12
df2 = df2.drop(['Date', 'y_test', 'y_pred', 'Div','FTAG','FTHG','H', 'A', 'D','Y','year', 'HomeTeam', 'AwayTeam'], axis=1)

# Replace outcomes to integer values
df2 = df2.replace(outcome_map)

X = df2.drop(['correct'], axis=1)
y = df2['correct']

# Load model from file 'model/xgb_model.sav'
xgb_model = pickle.load(open('data/models/xgb_global_filter.sav', 'rb'))

# Predict outcome
yy_pred = xgb_model.predict(X)

# Calculate accuracy for yy_pred
# accuracy = (yy_pred == y).sum() / len(y)
# print(accuracy)

df3 = pd.merge(df, pd.DataFrame({'yy_pred': yy_pred}), left_index=True, right_index=True)

# Drop all rows where yy_pred == 0
df3 = df3[df3['yy_pred'] == 1]

# Calculate accuracy y_test == 2 / rows count
# accuracy = (df3['y_test'] == 2).sum() / len(df3)
# print(accuracy)

# Calculate accuracy for correct
accuracy = (df3['correct'] == 1).sum() / len(df3)
print("Accuracy: ", accuracy)

# Merge df3 with original dataset
df4 = pd.merge(df3, dataset, left_index=True, right_index=True)

# Drop yy_pred
df4 = df4.drop(['yy_pred','Div','Date', 'x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'], axis=1)

# Add win column = (D - 1) if y_test == 2 else 0
df4['win'] = np.where(df4['y_test'] == 2, df4['D'] - 1, -1)

# Total bets
print(f'Total bets: {len(df4)}')

# Sum win column
total_win = df4['win'].sum()
print(f'Total win: {total_win}')

# Calculate ROI
roi = total_win / len(df4)
print(f'ROI: {roi}')

# Reset index
df4 = df4.reset_index(drop=True)

# Plot total win
df4['win'].cumsum().plot()
plt.show()