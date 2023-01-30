import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("data/good/position.csv")

# Convert all to integers
data = data.astype(int)

# Drop the columns that are not needed
X = data.drop(['Next Position','Changed'], axis=1)

print(X.head(50))

# Assign the input variables to X and the output variable to y
y = data['Changed']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_model.fit(X_train, y_train)  
y_pred = nn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)