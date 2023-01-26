import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Read in dataset
dataset = pd.read_csv(f"data/test.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

print(data)

# Get only data
X_test = data.drop(['Date','Div', 'HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)
y_test = data['Y']

print(X_test)

# Load model from file 'model/nn_model.sav'
knn_model = pickle.load(open('data/models/global_model_knn.sav', 'rb'))
y_pred = knn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# Print the accuracy of the model
print("Accuracy of the model: ", knn_model.score(X_test, y_test))