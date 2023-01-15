import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import process as p

for file in glob.glob('data/*.csv'):
    os.remove(file)

for file in glob.glob('data/*.png'):
    os.remove(file)

X_train, X_test, y_train, y_test, data_test = p.data()

for i in range(2):
    # Logistic Regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    p.process(y_test, y_pred, data_test, 'regression')

for i in range(2):
    # Decision Tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    p.process(y_test, y_pred, data_test, 'decision')

for i in range(2):
    # SVM model
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    p.process(y_test, y_pred, data_test, 'svm')

for i in range(2):
    # XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    p.process(y_test, y_pred, data_test, 'xgboost')

for i in range(2):
    # Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    p.process(y_test, y_pred, data_test, 'random')

for i in range(2):
    # KNN model
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    p.process(y_test, y_pred, data_test, 'knn')

for i in range(2):
    # Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    p.process(y_test, y_pred, data_test, 'naive')

for i in range(2):
    # Neural Network model
    nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
    nn_model.fit(X_train, y_train)
    y_pred = nn_model.predict(X_test)
    p.process(y_test, y_pred, data_test, 'nn')

for file in glob.glob('data/*.csv'):
    os.remove(file)

for file in glob.glob('data/*.png'):
    if 'accuracies' in file:
        os.remove(file)