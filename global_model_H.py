import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Read in dataset
dataset = pd.read_csv(f"data/train.csv", header=0)

print(f'Number of rows: {len(dataset)}')

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Add column called HH and set to 1 if Y == 3
data['HH'] = np.where(data['Y'] == 3, 1, 0)

# Set HH = 1 if Y == 4
data.loc[data['Y'] == 4, 'HH'] = 1

# Assign the input variables to X and the output variable to y
X = data.drop(['HH','Date','Div', 'HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)
y = data['HH']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
# Save model to file
# pickle.dump(xgb_model, open('data/models/xgb_model_global.sav', 'wb'))
y_pred = xgb_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Accuracy: {accuracy}')

# Merge y_pred and data
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

# Add column H from data
df['H'] = data['H']

# Add win column and set = (H - 1) if y_test == 1 else -1
df['win'] = np.where(df['y_test'] == 1, df['H'] - 1, -1)

# Drop all rows where y_pred == 0
# df = df[(df['y_pred'] != 0)]

# Reset index
df = df.reset_index(drop=True)

# Devide y_test == 1 / total
print(f'Percentage of H: {np.sum(df["y_test"] == 1) / len(df)}')

# Plot win column
df['win'].cumsum().plot()
# plt.show()

# Merge df and data
df2 = df.merge(data, left_index=True, right_index=True)
print(f'Number of rows: {len(df2)}')

# Drop all rows except y_pred, y_test, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12
df2 = df2.drop(['HH', 'H_x', 'H_y', 'win', 'Date','Div', 'HomeTeam','AwayTeam','FTAG','FTHG', 'A', 'D','Y'], axis=1)

X = df2.drop(['y_test', 'y_pred'], axis=1)
y = df2['y_test']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
# Save model to file
# pickle.dump(xgb_model, open('data/models/xgb_model_global.sav', 'wb'))
yy_pred = xgb_model.predict(X_test)
yy_pred = np.round(yy_pred).astype(int)

# Calculate accuracy
accuracy = np.sum(yy_pred == y_test) / len(y_test)
print(f'Accuracy: {accuracy}')

# Add column win and set = (H - 1) if y_test == 1 else -1
df2['win'] = np.where(df2['y_test'] == 1, df['H'] - 1, -1)

# Plot win column
df2['win'].cumsum().plot()
plt.show()