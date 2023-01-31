import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Read the data
train = pd.read_csv("data/titanic/train.csv")
test = pd.read_csv("data/titanic/test.csv")

# Drop the columns that are not needed
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
X = train.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
y = train['Survived']

print(X.head())

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

# Convert sex to int
# Create a dictionary to map outcome to integer values
outcome_map = {'male': 0, 'female': 1}

# Create a new column "outcome_num" to store the mapped outcome
X = X.replace(outcome_map)
X_test = X_test.replace(outcome_map)

X_train = X
# Assign the input variables to X and the output variable to y
y_train = train['Survived']

# Xgboost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print("XGBoost model")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))