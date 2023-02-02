import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the data
train = pd.read_csv("data/titanic/train.csv")

# Drop NA values
train = train.dropna()

# Drop the columns that are not needed
X = train.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
y = train['Survived']

replace_map = {'male':0, 'female':1}
X = X.replace(replace_map)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

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

# Merge predictions with original test data
df = pd.DataFrame({'test': y_test, 'Survived': y_pred})

# Use XGBoost
import xgboost as xgb

# Xgboost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print("XGBoost model")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

# Logistic regression
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