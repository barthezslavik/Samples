import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Read in dataset
dataset = pd.read_csv(f"data/train.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

# Get only data
X = data.drop(['Date','Div','HomeTeam','AwayTeam','FTAG','FTHG','H', 'A', 'D','Y'], axis=1)
y = data['Y']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
XX_test = X_test

print(f'Number of rows: {len(X_train)}')

# Use KNN to predict outcome = 2
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_test, y_test)
# Save model
pickle.dump(knn, open('data/models/global_model_knn.sav', 'wb'))
y_pred = knn.predict(X_test)

print(f'Number of predictions: {len(y_pred)}')

# Calculate accuracy for each outcome
for i in range(5):
    print(f'Accuracy for outcome {i}: {accuracy_score(y_test[y_test == i], y_pred[y_test == i])}')


knn = pickle.load(open('data/models/global_model_knn.sav', 'rb'))
y_pred = knn.predict(X_test)

print(f'Number of predictions: {len(y_pred)}')

# Calculate accuracy for each outcome
for i in range(5):
    print(f'Accuracy for outcome {i}: {accuracy_score(y_test[y_test == i], y_pred[y_test == i])}')
