import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("data/good/train.csv")

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = data.replace(outcome_map)

# Define the features and target
X = data[['H', 'D', 'A', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]
y = data['Y']

# Define the custom loss function
def custom_loss(y_true, y_pred):
    # DMatrix to DataFrame
    y_true = pd.DataFrame(y_true)
    print(y_pred)
    # odds = y_pred[:, :3]
    # bets = y_pred[:, 3:]
    # y_true = y_true.values
    
    # # calculate profits for each outcome
    # profits = np.zeros_like(y_true)
    # profits[y_true == 'BW'] = (bets[:, 0] * odds[:, 0]) - bets[:, 0]
    # profits[y_true == 'SW'] = (bets[:, 1] * odds[:, 0]) - bets[:, 1]
    # profits[y_true == 'D'] = (bets[:, 2] * odds[:, 1]) - bets[:, 2]
    # profits[y_true == 'BL'] = (bets[:, 3] * odds[:, 2]) - bets[:, 3]
    # profits[y_true == 'SL'] = (bets[:, 4] * odds[:, 2]) - bets[:, 4]
    
    # # return mean profit
    # return -np.mean(profits)

    return -1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_test)

# Define the model
params = {'objective': 'multi:softprob', 'num_class': 4}
xgb_model = xgb.XGBClassifier(**params)

# Fit the model with the custom loss function
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=custom_loss, verbose=True)

# Make predictions
y_pred = xgb_model.predict(X_test)

print(y_pred)
