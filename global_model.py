import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read in dataset
dataset = pd.read_csv(f"data/train.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Assign the input variables to X and the output variable to y
# Date,Div,HomeTeam,AwayTeam,FTAG,FTHG,H,D,A,Y,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12
X = data.drop(['Date','Div', 'HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)

# X = data.drop(['date','team1','team2','y'], axis=1)
# y = data['y']
y = data['Y']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_model.fit(X_train, y_train)
# Save model to file
pickle.dump(nn_model, open('data/models/nn_model_global.sav', 'wb'))   
y_pred = nn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
# Save model to file
pickle.dump(xgb_model, open('data/models/xgb_model_new.sav', 'wb'))
y_pred2 = xgb_model.predict(X_test)
y_pred2 = np.round(y_pred2).astype(int)

# Merge y_pred and y_pred2, if y_pred == 2, use y_pred2, else use y_pred
y_pred3 = np.where(y_pred == 2, y_pred2, y_pred)
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred3})
# Drop all rows where y_pred == 0 and y_pred == 4
df = df[(df['y_pred'] != 0) & (df['y_pred'] != 4)]
# Add column called correct and set to 1 if y_test == y_pred
df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)
# Drop all rows where y_pred != 3
df = df[(df['y_pred'] == 3)]

# Calculate accuracy
accuracy = df['correct'].sum() / df['correct'].count()
print(accuracy)