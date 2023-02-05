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

# Replace BL, SL, D, SW, BW
data = data.replace(['BW', 'SW', 'D', 'SL', 'BL'], [0, 0, 1, 2, 2])

# Add column Expected
# Set Expected = 1 if Y == 0 and H < A
# Set Expected = 1 if Y == 2 and A < H
# Set Expected = 0 otherwise
data['Expected'] = np.where((data['Y'] == 0) & (data['H'] < data['A']), 1, 0)
data['Expected'] = np.where((data['Y'] == 2) & (data['A'] < data['H']), 1, data['Expected'])

# Split data into features and labels
X = data.drop(['Y','Date','Home','Away', 'Expected'], axis=1)
y = data['Expected']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# # XGBoost model
# model = xgb.XGBClassifier()

# # Fit the model
# model.fit(X_train, y_train)

# # Make predictions for test data
# y_pred = model.predict(X_test)

# SVM model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions for test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Merge with original dataset
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
df = df.merge(data[['Date','Home','Away','H','D','A']], left_index=True, right_index=True)

# Merge Y from original dataset
df = df.merge(data[['Y']], left_index=True, right_index=True)

# Add column bet = ''
df['Bet'] = ''

# Set Bet = 0 if y_pred == 1 and H < A
df['Bet'] = np.where((df['y_pred'] == 1) & (df['H'] < df['A']), 0, df['Bet'])

# Set Bet = 2 if y_pred == 1 and A < H
df['Bet'] = np.where((df['y_pred'] == 1) & (df['A'] < df['H']), 2, df['Bet'])

# Set Bet = 0 if y_pred == 0 and H > A
df['Bet'] = np.where((df['y_pred'] == 0) & (df['H'] > df['A']), 0, df['Bet'])

# Set Bet = 2 if y_pred == 0 and A > H
df['Bet'] = np.where((df['y_pred'] == 0) & (df['A'] > df['H']), 2, df['Bet'])

# Drop all rows where y_pred == 0
df = df[df['Bet'] != 0]

# Add column Win = 0
df['Win'] = -1

# Set win = (H - 1) if Y == Bet and Bet == 0
df['Win'] = np.where((df['Y'] == df['Bet']) & (df['Bet'] == 0), df['H'] - 1, df['Win'])

# Set win = (D - 1) if Y == Bet and Bet == 1
df['Win'] = np.where((df['Y'] == df['Bet']) & (df['Bet'] == 2), df['A'] - 1, df['Win'])

print(df.head(50))

# Total bets
total_bets = df.shape[0]
print(f'Total bets: {total_bets}')

# Calculate profit
profit = df['Win'].sum()
print(f'Profit: {profit}')

# Calculate profit per game
profit_per_game = profit / len(df)
print(f'Profit per game: {profit_per_game}')