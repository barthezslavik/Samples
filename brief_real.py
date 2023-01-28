import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# Read in dataset
dataset = pd.read_csv(f"data/good/train.csv", header=0)

# Drop all rows where H, D, A equal NaN
dataset = dataset.dropna(subset=['H', 'D', 'A'])

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

print("Length of data: ", len(data))

# Get only data
X_test = data.drop(['Date','Div', 'HomeTeam','AwayTeam','FTAG','FTHG','Y'], axis=1)
y_test = data['Y']

print(X_test.head())

#xgb_model = xgb.XGBClassifier()

# Load model from file 'data/models/xgb_brief.sav'
#xgb_model.load_model('data/models/xgb_brief.sav')
xgb_model = pickle.load(open('data/models/xgb_brief.sav', 'rb'))
y_pred = xgb_model.predict(X_test)

# Replace BW -> SW, BL -> SL
y_pred[y_pred == 4] = 3
y_pred[y_pred == 0] = 1

# Print the accuracy for each outcome
print("Accuracy for each outcome:")
print("BL: ", np.sum(y_pred[y_test == 0] == 0) / np.sum(y_test == 0))
print("SL: ", np.sum(y_pred[y_test == 1] == 1) / np.sum(y_test == 1))
print("D: ", np.sum(y_pred[y_test == 2] == 2) / np.sum(y_test == 2))
print("SW: ", np.sum(y_pred[y_test == 3] == 3) / np.sum(y_test == 3))
print("BW: ", np.sum(y_pred[y_test == 4] == 4) / np.sum(y_test == 4))

# Merge the predictions with original data
data_pred = pd.DataFrame({'Y': y_test, 'Y_pred': y_pred})

# Merge with H, D, A
data_pred = data_pred.merge(data[['H', 'D', 'A']], left_index=True, right_index=True)

base_pred = data_pred

# Drop all rows where Y_pred == 3
data_pred = data_pred[data_pred['Y_pred'] == 3]

print("Total bets on H: ", len(data_pred))

# Add a column for the profit, set = (H - 1) if Y = 3 and -1 otherwise
data_pred['Profit'] = np.where(data_pred['Y'] == 3, data_pred['H'] - 1, -1)

# Calculate the total profit
print("Total profit H: ", data_pred['Profit'].sum())

# ROI
print("ROI H: ", (data_pred['Profit'].sum() / len(data_pred)) * 100)

data_pred = base_pred

# Drop all rows where Y_pred == 2
data_pred = data_pred[data_pred['Y_pred'] == 2]

print("Total bets on D: ", len(data_pred))

# Add a column for the profit, set = (H - 1) if Y = 3 and -1 otherwise
data_pred['Profit'] = np.where(data_pred['Y'] == 2, data_pred['D'] - 1, -1)

# Calculate the total profit
print("Total profit D: ", data_pred['Profit'].sum())

# ROI
print("ROI D: ", (data_pred['Profit'].sum() / len(data_pred)) * 100)

d_pred = data_pred

data_pred = base_pred

# Drop all rows where Y_pred == 1
data_pred = data_pred[data_pred['Y_pred'] == 1]

print("Total bets on A: ", len(data_pred))

# Add a column for the profit, set = (H - 1) if Y = 3 and -1 otherwise
data_pred['Profit'] = np.where(data_pred['Y'] == 1, data_pred['A'] - 1, -1)

# Calculate the total profit
print("Total profit A: ", data_pred['Profit'].sum())

# ROI
print("ROI A: ", (data_pred['Profit'].sum() / len(data_pred)) * 100)

# Calculate total number of bets
total_bets = np.sum(y_pred == 3) + np.sum(y_pred == 1) + np.sum(y_pred == 4) + np.sum(y_pred == 0) + np.sum(y_pred == 2)

# Calculate total number of bets for SW
sw_bets = np.sum(y_pred == 3)
print("Total number of bets for SW: ", sw_bets)

# Calculate total number of bets for SL
sl_bets = np.sum(y_pred == 1)
print("Total number of bets for SL: ", sl_bets)

# Calculate total number of bets for D
d_bets = np.sum(y_pred == 2)
print("Total number of bets for D: ", d_bets)

# Calculate total number of bets
total_bets = sw_bets + sl_bets + d_bets
print("Total number of bets: ", total_bets)

# Plot the profit
d_pred = d_pred.reset_index(drop=True)
plt.plot(d_pred['Profit'].cumsum())
# plt.show()