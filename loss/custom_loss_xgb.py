import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import xgboost as xgb
import numpy as np

# Define the custom loss function
def betting_profit_loss(y_true, y_pred, odds):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = - (y_true * np.log(y_pred) * odds + (1 - y_true) * np.log(1 - y_pred) * odds)
    return 'betting_profit', np.mean(loss), False

# Read data
data = pd.read_csv('data/good/fuzzy.csv')

# Replace BL, SL, D, SW, BW with 1, 2, 3, 4, 5
data = data.replace(['BL', 'SL', 'D', 'SW', 'BW'], [0, 0, 1, 2, 2])

# Split data into features and labels
# X = data.drop(['Y','Date','Home','Away'], axis=1) # -> 63% H, 37% A, 30% D
X = data.drop(['Y','Date','Home','Away','H','A'], axis=1) # -> 55% H, 41% D, 34% A
# X = data.drop(['Y','Date','Home','Away','H','D','A'], axis=1)
y = data['Y']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, tree_method='exact')

# Fit the model
model.fit(X_train, y_train)

# Make predictions for test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Show accuracy for each outcome
for i in range(3):
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    df = df[(df['y_pred'] == i)]
    df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)
    accuracy = df['correct'].sum() / df.shape[0]
    print(f'Accuracy for {i}: {accuracy}')