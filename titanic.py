import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data
train_data = pd.read_csv("data/titanic/train.csv")
test_data = pd.read_csv("data/titanic/test.csv")

columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
le = LabelEncoder()
columns = ['Sex', 'Embarked', 'ticket_type', 'cabin_type', 'title']

for col in columns:
    le.fit(train_data[col])
    train_data[col] = le.transform(train_data[col])
    
train_data.head()

# Drop the columns that are not needed
X = train_data.drop(['Survived'], axis=1)
X_test = test_data.drop(['PassengerId'], axis=1)

# Assign the input variables to X and the output variable to y
y = train_data['Survived']

# split data into train and test set
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=0.3)

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred1 = xgb_model.predict(X_val)

# Calculate accuracy for each of outcome
for i in range(2):
    accuracy = accuracy_score(y_val[y_val == i], y_pred1[y_val == i])
    print("Accuracy for outcome {}: {}".format(i, accuracy))