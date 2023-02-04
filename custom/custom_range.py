import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split

country = 'custom'

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
dataset = pd.read_csv(f"data/fuzzy/fuzzy5.csv", header=0)

# Remove all rows where 
# y == 'BL', 'SL' and 'A' not in range 2.7 - 3.8
# y == 'D' and 'D' not in range 2.7 - 3.8
# y == 'SW', 'BW' and 'H' not in range 2.7 - 3.8
# dataset = dataset[(dataset['y'] != 'BL') | ((dataset['y'] == 'BL') & (dataset['A'] >= 2.7) & (dataset['A'] <= 3.8))]
# dataset = dataset[(dataset['y'] != 'SL') | ((dataset['y'] == 'SL') & (dataset['A'] >= 2.7) & (dataset['A'] <= 3.8))]
# dataset = dataset[(dataset['y'] != 'D') | ((dataset['y'] == 'D') & (dataset['D'] >= 2.7) & (dataset['D'] <= 3.8))]
# dataset = dataset[(dataset['y'] != 'SW') | ((dataset['y'] == 'SW') & (dataset['H'] >= 2.7) & (dataset['H'] <= 3.8))]
# dataset = dataset[(dataset['y'] != 'BW') | ((dataset['y'] == 'BW') & (dataset['H'] >= 2.7) & (dataset['H'] <= 3.8))]

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Assign the input variables to X and the output variable to y
X = data.drop(['div','outcome','date','team1','team2','y', 'prediction', 'home_score', 'away_score','H','D','A'], axis=1)
y = data['y']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

print(X_train.head())

# Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_model.fit(X_train, y_train)
# Save model to file
pickle.dump(nn_model, open('data/models/nn_model_new.sav', 'wb'))
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