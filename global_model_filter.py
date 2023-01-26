import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read in dataset
dataset = pd.read_csv(f"data/new_global_prediction_train.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Assign the input variables to X and the output variable to y
X = data.drop(['correct', 'y_test','y_pred', 'win', 'coef', 'Date', 'Div', 'year', 'HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)
y = data['correct']

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X, y)
# Save model to file
pickle.dump(xgb_model, open('data/models/xgb_global_filter.sav', 'wb'))
y_pred = xgb_model.predict(X)
y_pred = np.round(y).astype(int)

# Accuracy y_pred
df = pd.DataFrame({'y_test': y, 'y_pred': y_pred})
# Add column called correct and set to 1 if y_test == y_pred
df['xcorrect'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

# Calculate accuracy
accuracy = df['xcorrect'].sum() / df['xcorrect'].count()
print(f'Accuracy: {accuracy}')