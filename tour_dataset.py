import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential

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

# Replace the team names with numbers
teams = df["HomeTeam"].unique()
teams.sort()
teams_dict = {}
for i, team in enumerate(teams):
    teams_dict[team] = i
df["HomeTeam"] = df["HomeTeam"].apply(lambda x: teams_dict[x])
df["AwayTeam"] = df["AwayTeam"].apply(lambda x: teams_dict[x])

# Drop all the columns except FTHG, FTAG, HomePoints, AwayPoints, Tour
df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HomePoints", "AwayPoints", "Tour"]]

# Define the number of timesteps
timesteps = 3

# Create the input dataset
X_train = np.zeros((df.shape[0] - timesteps, timesteps, 5))
for i in range(timesteps, df.shape[0]):
    X_train[i - timesteps] = df.iloc[i - timesteps:i, [0, 1, 2, 3, 6]].values

# Create the output dataset
y_train = df.iloc[timesteps:, [4, 5]].values

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# Create input data for the next tour
x_next_tour = df.iloc[-timesteps:, [0, 1, 2, 3, 6]].values
x_next_tour = np.reshape(x_next_tour, (1, x_next_tour.shape[0], x_next_tour.shape[1]))

# Make the prediction
y_pred = model.predict(x_next_tour)

print(y_pred)