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

# Calculate accuracy by single outcome SW
acc_sw = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])
print("Accuracy SW: ", acc_sw)

# Calculate accuracy by single outcome SL
acc_sl = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
print("Accuracy SL: ", acc_sl)

# Calculate accuracy by single outcome D
acc_d = accuracy_score(y_test[y_test == 2], y_pred[y_test == 2])
print("Accuracy D: ", acc_d)

# Calculate accuracy by single outcome BW
acc_bw = accuracy_score(y_test[y_test == 3], y_pred[y_test == 3])
print("Accuracy BW: ", acc_bw)

# Calculate accuracy by single outcome BL
acc_bl = accuracy_score(y_test[y_test == 4], y_pred[y_test == 4])
print("Accuracy BL: ", acc_bl)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# Save all accuracies to a file
with open('data/accuracies_d.csv', 'a') as f:
    data = [acc_d, acc]
    # Convert list to string
    data = ','.join(map(str, data)) + "\n"
    f.write(data)


# Plot accuracies.csv
accuracies = pd.read_csv('data/accuracies_d.csv', header=None)
accuracies.columns = ['D', 'Overall']
# Plot mean accuracy
# accuracies.mean().plot(kind='bar')
# plt.xlabel('Outcome')
# plt.ylabel('Accuracy')
# plt.savefig('data/mean_accuracy.png')
# Plot all accuracies
accuracies.plot()
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.savefig('data/accuracies.png')


# Plot the loss
plt.plot(nn.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
# plt.show()

y_test = y_test.replace({v: k for k, v in outcome_map.items()})
# print(y_test.head(20))
y_pred = pd.Series(y_pred).replace({v: k for k, v in outcome_map.items()})
# print(y_pred.head(20))

# Dump the test set to a file
y_test.to_csv('data/y_test.csv', index=False)
# Dump the predicted values to a file
y_pred.to_csv('data/y_pred.csv', index=False)