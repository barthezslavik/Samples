import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv("data/fuzzy/fuzzy5.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = data.replace(outcome_map)

# Add column correct if y == y_pred
data['correct'] = np.where(data['y'] == data['y_pred'], 1, 0)

X = data.drop(['div', 'outcome', 'date','team1','team2','home_score','away_score','y', 'correct'], axis=1)
y = data['correct']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model on the training data
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Use the trained model to predict the outcomes on the test data
y_pred = clf.predict(X_test)

# Convert the predictions to a dataframe
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

# Add column called correct and set to 1 if y_test == y_pred
df['confidence'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

# Devide correct by total rows
print(df['confidence'].sum() / df['confidence'].count())