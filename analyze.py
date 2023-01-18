import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import process as p

for file in glob.glob('data/*.png'):
    os.remove(file)

for file in glob.glob('data/mean/*.csv'):
    os.remove(file)

with open(f"data/mean/nn.csv", 'a') as f:
    f.write("Country,Year,Period,D,SL,BL,SW,BW,Overall\n")

with open(f"data/mean/xgboost.csv", 'a') as f:
    f.write("Country,Year,Period,D,SL,BL,SW,BW,Overall\n")

max_range = 2
years = [1996, 2000, 2004, 2008, 2012]
countries = ['belgium', 'germany']
periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for year in years:
    for country in countries:
        for period in periods:
            print(country, year, period)
            for file in glob.glob('data/*.csv'):
                os.remove(file)

            # XGBoost model
            for i in range(max_range):
                X_train, X_test, y_train, y_test, test_data = p.data(country, period, year)
                if len(X_train) == 0:
                    continue
                xgb_model = xgb.XGBClassifier()
                xgb_model.fit(X_train, y_train)
                y_pred = xgb_model.predict(X_test)
                p.process(y_test, y_pred, 'xgboost', test_data)
            p.plot_mean('xgboost', country, year, period)

            # Neural Network model
            for i in range(max_range):
                X_train, X_test, y_train, y_test, test_data = p.data(country, period, year)
                if len(X_train) == 0:
                    continue
                nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
                nn_model.fit(X_train, y_train)
                y_pred = nn_model.predict(X_test)
                p.process(y_test, y_pred, 'nn', test_data)
            p.plot_mean('nn', country, year, period)

            for file in glob.glob('data/*.csv'):
                os.remove(file)

# for file in glob.glob('data/*.png'):
#     os.remove(file)