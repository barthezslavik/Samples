from sklearn.ensemble import RandomForestClassifier
import process as p

X_train, X_test, y_train, y_test, data_test = p.data()

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

p.process(y_test, y_pred, data_test, 'random')