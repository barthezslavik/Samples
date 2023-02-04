import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split

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
dataset = pd.read_csv(f"data/stacking/dataset.csv", header=0)
# dataset = pd.read_csv(f"data/train.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

print(data.head())

# Assign the input variables to X and the output variable to y
X = data.drop(['y'], axis=1)
y = data['y']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Neural Network model
# nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
# nn_model.fit(X_train, y_train)
# # Save model to file
# pickle.dump(nn_model, open('data/models/nn_model_stacking.sav', 'wb'))   
# y_pred = nn_model.predict(X_test)
# y_pred = np.round(y_pred).astype(int)

# Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
# Save model to file
# pickle.dump(lr_model, open('data/models/lr_model_stacking.sav', 'wb'))
y_pred = lr_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

print(y_pred)