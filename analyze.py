import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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

for file in glob.glob('data/*.csv'):
    os.remove(file)