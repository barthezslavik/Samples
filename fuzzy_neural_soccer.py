import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read in dataset
data = pd.read_csv('data/fuzzy/fuzzy3.csv', header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'SW': 0, 'SL': 1, 'D': 2, 'BW': 3, 'BL': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = data.replace(outcome_map)

# Assign the input variables to X and the output variable to y
X = data.drop(['date','team1','team2','y'], axis=1)
y = data['y']

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the neural network
nn = MLPRegressor(hidden_layer_sizes=(20, 20))
nn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nn.predict(X_test)

# Convert predictions to integer values
y_pred = np.round(y_pred).astype(int)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

# Print accuracy
print("Accuracy: ", acc)

# Plot the loss
plt.plot(nn.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
# plt.show()

# Replace all values in X_test to their corresponding outcome
X_test = X_test.replace({v: k for k, v in outcome_map.items()})
print(X_test.head(20))
y_test = y_test.replace({v: k for k, v in outcome_map.items()})
print(y_test.head(20))
y_pred = pd.Series(y_pred).replace({v: k for k, v in outcome_map.items()})
print(y_pred.head(20))

# Dump the test set to a file
X_test.to_csv('data/X_test.csv', index=False)
# Dump the test set to a file
y_test.to_csv('data/y_test.csv', index=False)
# Dump the predicted values to a file
y_pred.to_csv('data/y_pred.csv', index=False)