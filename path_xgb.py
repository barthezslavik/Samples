import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("data/good/position_2013.csv")

# Convert all to integers
data = data.astype(int)

# Drop the columns that are not needed
X = data.drop(['Next Position','Changed','Next Points'], axis=1)

# Assign the input variables to X and the output variable to y
y = data['Changed']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred1 = xgb_model.predict(X_test)

print("XGBoost model")

# Calculate accuracy for each of outcome
for i in range(3):
    accuracy = accuracy_score(y_test[y_test == i], y_pred1[y_test == i])
    print("Accuracy for outcome {}: {}".format(i, accuracy))

# Decision Tree model
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred2 = dt_model.predict(X_test)

print("Decision Tree model")

# Calculate accuracy for each of outcome
for i in range(3):
    accuracy = accuracy_score(y_test[y_test == i], y_pred2[y_test == i])
    print("Accuracy for outcome {}: {}".format(i, accuracy))

# Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred3 = lr_model.predict(X_test)

print("Logistic Regression model")

# Calculate accuracy for each of outcome
for i in range(3):
    accuracy = accuracy_score(y_test[y_test == i], y_pred3[y_test == i])
    print("Accuracy for outcome {}: {}".format(i, accuracy))

# Merge y_pred and y_test
# df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

# Save the data to a csv file
# df.to_csv('data/predictions/next_tour.csv', index=False)

# # Merge y_pred and y_test
# df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})