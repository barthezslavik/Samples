import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("data/tour/E0.csv")

# Create new columns for the points
df["HomePoints"] = 0
df["AwayPoints"] = 0

# Create a dict to store the accumulated points
points = {}

# Iterate over the rows of the dataset
for index, row in df.iterrows():
    # Check the result of the match
    if row["FTHG"] > row["FTAG"]:
        # Home team wins
        points[row["HomeTeam"]] = points.get(row["HomeTeam"], 0) + 3
    elif row["FTHG"] == row["FTAG"]:
        # Draw
        points[row["HomeTeam"]] = points.get(row["HomeTeam"], 0) + 1
        points[row["AwayTeam"]] = points.get(row["AwayTeam"], 0) + 1
    else:
        # Away team wins
        points[row["AwayTeam"]] = points.get(row["AwayTeam"], 0) + 3
    df.at[index, "HomePoints"] = points.get(row["HomeTeam"], 0)
    df.at[index, "AwayPoints"] = points.get(row["AwayTeam"], 0)

# Create a new column for the tour
df["Tour"] = 0

# Create a dict to store the current tour for each team
current_tour = {}

# Iterate over the rows of the dataset
for index, row in df.iterrows():
    current_tour[row["HomeTeam"]] = current_tour.get(row["HomeTeam"], 0) + 1
    df.at[index, "Tour"] = current_tour[row["HomeTeam"]]

# Drop all the columns except FTHG, FTAG, HomePoints, AwayPoints, Tour
df = df[["HomeTeam", "AwayTeam", "HomePoints", "AwayPoints", "Tour"]]

# Save the dataset
df.to_csv("data/tour/E0_1.csv", index=False)

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# convert categorical columns to numerical values
le = LabelEncoder()
df['HomeTeam'] = le.fit_transform(df['HomeTeam'])
df['AwayTeam'] = le.fit_transform(df['AwayTeam'])

# split the data into training and testing sets
X = df.drop(['HomePoints', 'AwayPoints'], axis=1)
y = df[['HomePoints', 'AwayPoints']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create the XGBoost model
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

# fit the model to the training data
xg_reg.fit(X_train, y_train)

# make predictions on the test set
predictions = xg_reg.predict(X_test)

# evaluate the model's performance
print("R2 score: ", r2_score(y_test, predictions))

# predict the HomePoints, AwayPoints for a specific HomeTeam, AwayTeam for the next tour
team1 = 'Brentford'
team2 = 'Arsenal'
team1_encoded = le.transform([team1])[0]
team2_encoded = le.transform([team2])[0]
next_tour_data = np.array([[team1_encoded, team2_encoded, 2]])
predictions = xg_reg.predict(next_tour_data)
print(predictions)

# print("Predicted HomePoints: ", predictions[0][0])
# print("Predicted AwayPoints: ", predictions[0][1])

# # Decoding the team names for X_test
# X_test['HomeTeam'] = le.inverse_transform(X_test['HomeTeam'])
# X_test['AwayTeam'] = le.inverse_transform(X_test['AwayTeam'])

# # Merge the predictions with the X_test
# X_test['HomePoints'] = predictions[:,0]
# X_test['AwayPoints'] = predictions[:,1]

# print(X_test)