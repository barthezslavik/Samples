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

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Get only data
X_test = data.drop(['Date','Div', 'year', 'HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)
y_test = data['Y']

print(f'Number of rows: {len(X_test)}')

# Load model from file 'model/xgb_model.sav'
xgb_model = pickle.load(open('data/models/xgb_global_model_filter.sav', 'rb'))

# Predict outcome
yy_pred = xgb_model.predict(X_test)

df3 = pd.merge(data, pd.DataFrame({'yy_pred': yy_pred}), left_index=True, right_index=True)

# Add column correct = 1 if Y == 2
df3['correct'] = np.where(df3['Y'] == 2, 1, 0)

# Drop all rows where yy_pred == 0
df3 = df3[df3['yy_pred'] == 1]

# Print number of rows
print(f'Number of rows: {len(df3)}')

# # Calculate accuracy for correct
accuracy = (df3['correct'] == 1).sum() / len(df3)
print("Accuracy: ", accuracy)