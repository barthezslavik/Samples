import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("data/good/train.csv")

# Drop all rows where H < 2 or A < 2
data = data[(data['H'] >= 2) & (data['A'] >= 2)]

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