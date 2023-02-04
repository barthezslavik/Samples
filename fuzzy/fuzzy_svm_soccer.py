from sklearn.svm import SVC
import process as p

X_train, X_test, y_train, y_test, data_test = p.data()

# Train the SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

p.process(y_test, y_pred, data_test, 'svm')