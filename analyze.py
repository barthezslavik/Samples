import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import process as p

max_range = 2
country = 'belgium'
learn_period = 3
start_year = 2010

for file in glob.glob('data/*.csv'):
    os.remove(file)

for file in glob.glob('data/*.png'):
    os.remove(file)

for i in range(max_range):
    X_train, X_test, y_train, y_test, test_data = p.data(country, learn_period, start_year)
    # XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    p.process(y_test, y_pred, 'xgboost', test_data)
p.plot_mean('xgboost', country)

for i in range(max_range):
    X_train, X_test, y_train, y_test, test_data = p.data(country, learn_period, start_year)
    # Neural Network model
    nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
    nn_model.fit(X_train, y_train)
    y_pred = nn_model.predict(X_test)
    p.process(y_test, y_pred, 'nn', test_data)
p.plot_mean('nn', country)

for file in glob.glob('data/*.csv'):
    os.remove(file)