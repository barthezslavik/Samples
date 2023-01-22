import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv("data/fuzzy/fuzzy5.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = data.replace(outcome_map)

# Add column correct if y == y_pred
data['correct'] = np.where(data['y'] == data['prediction'], 1, 0)

# drop columns
drop_columns = ['div', 'outcome', 'date','team1','team2','home_score','away_score','y', 'correct']
# 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']

X = data.drop(drop_columns, axis=1)
y = data['correct']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
clf = XGBClassifier()
clf.fit(X_train, y_train)

# Use the trained model to predict the outcomes on the test data
y_pred = clf.predict(X_test)

print("Accuracy: ", np.sum(y_pred == y_test) / len(y_test))

# Merge y_pred with y_test
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

# Get the index of the selected rows
index = df.index

# Get the rows from the original dataset
df = data.iloc[index]

# Merge y_pred with original dataset
df = pd.merge(df, pd.DataFrame({'y_pred': y_pred}), left_index=True, right_index=True)

# Drop all rows where y_pred != 1
df = df[df['y_pred'] == 0]

# Drop all rows where prediction == 2
df = df[df['prediction'] != 2]

# Calculate accuracy for correct column
accuracy = df['correct'].sum() / df.shape[0]
print(f'Accuracy: {accuracy}')

# Save to csv
df.to_csv('data/fuzzy/predictions.csv', index=False)