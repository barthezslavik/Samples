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
X = data.drop(['y'], axis=1)
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
plt.show()