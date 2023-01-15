from sklearn.tree import DecisionTreeClassifier
import process as p

X_train, X_test, y_train, y_test, data_test = p.data()

# Train the Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

p.process(y_test, y_pred, data_test, 'decision')