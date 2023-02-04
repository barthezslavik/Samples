import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Read the data
train = pd.read_csv("data/titanic/train.csv")
test = pd.read_csv("data/titanic/test.csv")

# Drop NA values
train = train.dropna()

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

# Use neural network
from sklearn.neural_network import MLPClassifier

# Create a neural network
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

# Train the model
mlp.fit(X_train, y_train)

# Predict the outcome
y_pred = mlp.predict(X_test)

print("Neural network model")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

# Use logistic regression
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)

# Predict the outcome
y_pred = logreg.predict(X_test)

print("Logistic regression model")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

# Support Vector Machines
from sklearn.svm import SVC

# Create a support vector classifier
svc = SVC()

# Train the model
svc.fit(X_train, y_train)

# Predict the outcome
y_pred = svc.predict(X_test)

print("Support vector machine model")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

# Use decision tree
from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
dtree = DecisionTreeClassifier()

# Train the model
dtree.fit(X_train, y_train)

# Predict the outcome
y_pred = dtree.predict(X_test)

print("Decision tree model")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))