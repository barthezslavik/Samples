import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read in dataset
dataset = pd.read_csv(f"data/new_global_prediction_test.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Print dataset length
print(f'Dataset length: {len(data)}')

# Assign the input variables to X and the output variable to y
X = data.drop(['correct', 'y_test','y_pred', 'win', 'coef', 'Date', 'Div', 'year', 'HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)
y = data['correct']

# Load model from file 'model/xgb_model.sav'
xgb_model = pickle.load(open('data/models/xgb_global_filter.sav', 'rb'))
y_pred = xgb_model.predict(X)
y_pred = np.round(y_pred).astype(int)

# Accuracy y_pred
df = pd.DataFrame({'y_test': y, 'y_pred': y_pred})
# Add column called correct and set to 1 if y_test == y_pred
df['ycorrect'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

# Print y_pred = 1 count
print(f'Predictions: {df["y_pred"].sum()}')

# Calculate accuracy
accuracy = df['ycorrect'].sum() / df['ycorrect'].count()
print(f'Accuracy: {accuracy}')

# Save predictions to csv
df.to_csv('data/filter_prediction.csv', index=False)