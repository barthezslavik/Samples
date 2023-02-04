from sklearn.neural_network import MLPRegressor
import process as p

X_train, X_test, y_train, y_test, data_test = p.data()

# Train the neural network
nn = MLPRegressor(hidden_layer_sizes=(20, 20))
nn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nn.predict(X_test)

p.process(y_test, y_pred, data_test, 'nn')