import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import xgboost as xgb
import numpy as np

# Read data
data = pd.read_csv('data/good/fuzzy.csv')

# Replace BL, SL, D, SW, BW with 1, 2, 3, 4, 5
data = data.replace(['BW', 'SW', 'D', 'SL', 'BL'], [0, 0, 1, 2, 2])

# Replace E0, E1, E2 with 1, 2, 3
data = data.replace(['E0', 'E1', 'E2'], [1, 2, 3])

# Split data into features and labels
X = data.drop(['Y','Date','Home','Away'], axis=1)
y = data['Y']

print(X.head())

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# XGBoost model
model = xgb.XGBClassifier()

# Fit the model
model.fit(X_train, y_train)

# Make predictions for test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

accuracies = []

# Show accuracy for each outcome
for i in range(3):
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    df = df[(df['y_pred'] == i)]
    df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)
    accuracy = df['correct'].sum() / df.shape[0]
    accuracies.append(accuracy)
    print(f'Accuracy for {i}: {accuracy}')

k = 0
H_min = 1/accuracies[0] + k
D_min = 1/accuracies[1] + k
A_min = 1/accuracies[2] + k

print(f'H_min: {H_min}')
print(f'D_min: {D_min}')
print(f'A_min: {A_min}')

# Merge with original dataset
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
df = df.merge(data[['Date','Home','Away','H','D','A']], left_index=True, right_index=True)

# Add column called correct and set to 1 if y_test == y_pred
df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)

# Add column win = -1
df['win'] = 0

# Set win = (H - 1) if y_pred == 0 and H > H_min else -1
df.loc[(df['y_pred'] == 0) & (df['H'] > H_min), 'win'] = df['H'] - 1

# Set win = (D - 1) if y_pred == 1 and D > D_min else -1
df.loc[(df['y_pred'] == 1) & (df['D'] > D_min), 'win'] = df['D'] - 1

# Set win = (A - 1) if y_pred == 2 and A > A_min else -1
df.loc[(df['y_pred'] == 2) & (df['A'] > A_min), 'win'] = df['A'] - 1

# Set win = -1 if correct == 0
df.loc[df['correct'] == 0, 'win'] = -1

# Set win = 0 if y_pred = 0 and H < H_min
df.loc[(df['y_pred'] == 0) & (df['H'] < H_min), 'win'] = 0

# Set win = 0 if y_pred = 1 and D < D_min
df.loc[(df['y_pred'] == 1) & (df['D'] < D_min), 'win'] = 0

# Set win = 0 if y_pred = 2 and A < A_min
df.loc[(df['y_pred'] == 2) & (df['A'] < A_min), 'win'] = 0

print(df.tail(20))

# Total matches
total_matches = data.shape[0]
print(f'Total matches: {total_matches}')

# Drop rows where win == 0
df = df[df['win'] != 0]

# Total bets
total_bets = df.shape[0]
print(f'Total bets: {total_bets}')

# Calculate profit
profit = df['win'].sum()
print(f'Profit: {profit}')

# Profit per bet
profit_per_bet = df['win'].sum() / df.shape[0]
print(f'ROI: {profit_per_bet}')