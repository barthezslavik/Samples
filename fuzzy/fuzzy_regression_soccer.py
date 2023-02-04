from sklearn.linear_model import LogisticRegression
import process as p

X_train, X_test, y_train, y_test, data_test = p.data()

# Train the Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

p.process(y_test, y_pred, data_test, 'regression')