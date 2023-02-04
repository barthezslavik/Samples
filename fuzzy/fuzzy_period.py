import xgboost as xgb
import process as p

X_train, X_test, y_train, y_test, data_test = p.data()

# Train the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

p.process(y_test, y_pred, data_test, 'xgboost')