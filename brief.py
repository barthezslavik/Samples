import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("data/good/train.csv")

print("Length of data: ", len(data))

# Drop all rows where H < 2 or A < 2
data = data[(data['H'] >= 2) & (data['A'] >= 2)] # -> H 
# data = data[(data['H'] >= 1.6) & (data['A'] >= 1.6)] # -> H 
# data = data[(data['H'] >= 2.4) & (data['A'] >= 2.4)] # -> D

print("Length of data: ", len(data))

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = data.replace(outcome_map)

# Define the features and target
X = data[['H', 'D', 'A', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]
y = data['Y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier()

# Fit the model with the custom loss function
xgb_model.fit(X_train, y_train, verbose=True)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Replace BW -> SW, BL -> SL
y_pred[y_pred == 4] = 3
y_pred[y_pred == 0] = 1

# Calculate the accuracy for each outcome
acc = np.zeros(5)
for i in range(5):
    acc[i] = np.mean(y_pred[y_test == i] == y_test[y_test == i])
    print(f"Accuracy for outcome {i}: {acc[i]}")

# Merge the predictions with original data
data_pred = pd.DataFrame({'Y': y_test, 'Y_pred': y_pred})

# Replace the Y = 4 and Y = 0 with 3 and 1
data_pred['Y'][data_pred['Y'] == 4] = 3
data_pred['Y'][data_pred['Y'] == 0] = 1

# Merge with H, D, A
data_pred = data_pred.merge(data[['H', 'D', 'A']], left_index=True, right_index=True)

# Drop all rows where Y_pred != 3
data_pred = data_pred[data_pred['Y_pred'] == 3]

print("Length of data: ", len(data_pred))

# Add a column for the profit, set = (H - 1) if Y = 3 and -1 otherwise
data_pred['Profit'] = np.where(data_pred['Y'] == 3, data_pred['H'] - 1, -1)

print(data_pred)

# Calculate the total profit
print("Total profit: ", data_pred['Profit'].sum())