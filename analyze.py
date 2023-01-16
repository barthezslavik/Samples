import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import process as p

max_range = 50

for file in glob.glob('data/*.csv'):
    os.remove(file)

for file in glob.glob('data/*.png'):
    os.remove(file)

# for i in range(max_range):
#     X_train, X_test, y_train, y_test = p.data()
#     # Logistic Regression model
#     lr_model = LogisticRegression()
#     lr_model.fit(X_train, y_train)
#     y_pred = lr_model.predict(X_test)
#     p.process(y_test, y_pred, 'regression')
# p.plot_mean('regression')

# for i in range(max_range):
#     X_train, X_test, y_train, y_test = p.data()
#     # Decision Tree model
#     dt_model = DecisionTreeClassifier()
#     dt_model.fit(X_train, y_train)
#     y_pred = dt_model.predict(X_test)
#     p.process(y_test, y_pred, 'decision')
# p.plot_mean('decision')

for i in range(max_range):
    X_train, X_test, y_train, y_test = p.data()
    # XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    p.process(y_test, y_pred, 'xgboost')
p.plot_mean('xgboost')

# for i in range(max_range):
#     X_train, X_test, y_train, y_test = p.data()
#     # Random Forest model
#     rf_model = RandomForestClassifier()
#     rf_model.fit(X_train, y_train)
#     y_pred = rf_model.predict(X_test)
#     p.process(y_test, y_pred, 'random')
# p.plot_mean('random')

for i in range(max_range):
    X_train, X_test, y_train, y_test = p.data()
    # Neural Network model
    nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
    nn_model.fit(X_train, y_train)
    y_pred = nn_model.predict(X_test)
    p.process(y_test, y_pred, 'nn')
p.plot_mean('nn')

for file in glob.glob('data/*.csv'):
    os.remove(file)

# for file in glob.glob('data/*.png'):
#     if 'accuracies' in file:
#         os.remove(file)