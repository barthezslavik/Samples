import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import pickle

def format_date(date):
    date = date.replace("-", "/")
    date = date.split("/")
    if len(date[2]) == 2:
        if int(date[2]) < 20:
            date[2] = "20" + date[2]
        else:
            date[2] = "19" + date[2]
    return "-".join(date)

# Read in dataset
dataset = pd.read_csv(f"data/171000_train.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Get only data
X_test = data.drop(['Date','Div', 'HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)
y_test = data['Y']

# Load model from file 'model/nn_model.sav'
nn_model = pickle.load(open('data/models/nn_model_global.sav', 'rb'))
y_pred = nn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Load model from file 'model/xgb_model.sav'
xgb_model = pickle.load(open('data/models/xgb_model_global.sav', 'rb'))
y_pred2 = xgb_model.predict(X_test)
y_pred2 = np.round(y_pred2).astype(int)

# Merge y_pred and y_pred2, if y_pred == 3, use y_pred2, else use y_pred
y_pred3 = np.where(y_pred == 2, y_pred2, y_pred)
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred3})
# Drop all rows where y_pred == 0 and y_pred == 4
df = df[(df['y_pred'] != 0) & (df['y_pred'] != 4)]

# Replace predicted outcome with value from outcome_map
df = df.replace({v: k for k, v in outcome_map.items()})

# Add column called correct and set to 1 if y_test == y_pred
df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)
# Devide correct by total rows
accuracy = df['correct'].sum() / df.shape[0]
print(f'Accuracy: {accuracy}')

# Add column win = -1
df['win'] = -1

# Remove all except y_pred = D
df = df[df['y_pred'] == 'D']

# Add column coef and set to D from original dataset
df['coef'] = dataset['D']

# Remove all except y_pred = SL
# df = df[df['y_pred'] == 'SL']

# # Add column coef and set to A from original dataset
# df['coef'] = dataset['A']


# If correct == 1, set win = (coef - 1)
df.loc[df['correct'] == 1, 'win'] = df['coef'] - 1

# Sum win column
total_win = df['win'].sum()
print(f'Total win: {total_win}')

accuracy = df['correct'].sum() / df.shape[0]
print(f'Accuracy: {accuracy}')

# Calculate roi
roi = total_win / df.shape[0]
print(f'ROI: {roi}')

# Save data to csv
df.to_csv('data/global_prediction.csv', index=False)