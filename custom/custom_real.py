import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import pickle

country = 'fuzzy'

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
dataset = pd.read_csv(f"data/{country}/fuzzy3.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Get only data
X_test = data.drop(['date','team1','team2','y'], axis=1)
y_test = data['y']

print(X_test.shape)

# Load model from file 'model/nn_model.sav'
nn_model = pickle.load(open('data/models/nn_model_new.sav', 'rb'))
y_pred = nn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Load model from file 'model/xgb_model.sav'
xgb_model = pickle.load(open('data/models/xgb_model_new.sav', 'rb'))
y_pred2 = xgb_model.predict(X_test)
y_pred2 = np.round(y_pred2).astype(int)

# Merge y_pred and y_pred2, if y_pred == 2, use y_pred2, else use y_pred
y_pred3 = np.where(y_pred == 2, y_pred2, y_pred)
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred3})
# Drop all rows where y_pred == 0 and y_pred == 4
df = df[(df['y_pred'] != 0) & (df['y_pred'] != 4)]
# Add column called correct and set to 1 if y_test == y_pred
df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)
# Devide correct by total rows
# accuracy = df['correct'].sum() / df.shape[0]
# print(f'Accuracy: {accuracy}')

# Show accuracy for each outcome
for i in range(5):
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred3})
    df = df[(df['y_pred'] == i)]
    df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)
    accuracy = df['correct'].sum() / df.shape[0]
    # Replace i with outcome
    i = list(outcome_map.keys())[list(outcome_map.values()).index(i)]
    print(f'Accuracy for {i}: {accuracy}')

# Add y_pred to original dataset
dataset['prediction'] = y_pred3
# Map y_pred back to outcome
dataset = dataset.replace({v: k for k, v in outcome_map.items()})
# Save to csv
dataset.to_csv(f"data/{country}/fuzzy4.csv", index=False)