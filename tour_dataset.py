import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Load the dataset
df = pd.read_csv("data/tour/E0.csv")

# Create new columns for the points
df["HomePoints"] = 0
df["AwayPoints"] = 0
# Home team goals
df["HomeTeamGoals"] = 0
# Away team goals
df["AwayTeamGoals"] = 0

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

# Drop all the columns except FTHG, FTAG, HomePoints, AwayPoints, Tour
df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HomePoints", "AwayPoints"]]

# Save the dataset
df.to_csv("data/tour/E0_1.csv", index=False)

# Transorm Date to_datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Sort by Date
df = df.sort_values(by=['Date'])

# Drop the unnecessary columns
df = df.drop(["FTHG", "FTAG"], axis=1)

# Create a new dataset with points for each team
df_points = pd.DataFrame(columns=["Date", "Team", "Points"])

# Iterate over the rows of the dataset
for index, row in df.iterrows():
    # Create a new row for the home team
    df_points = df_points.append({
        "Date": row["Date"],
        "Team": row["HomeTeam"],
        "Points": row["HomePoints"]
    }, ignore_index=True)
    # Create a new row for the away team
    df_points = df_points.append({
        "Date": row["Date"],
        "Team": row["AwayTeam"],
        "Points": row["AwayPoints"]
    }, ignore_index=True)

# Group by Date and Team
df_points = df_points.groupby(["Team","Date"]).sum().reset_index()

df = df_points

data = df[['Points']].values

# create a function to split the data into input and output sequences
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# choose number of previous values to use as input
n_steps = 5

# split the data into input and output sequences
X, y = split_sequence(data, n_steps)

# reshape the input data
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit the model to the data
model.fit(X, y, epochs=200, verbose=1)

# make predictions
x_input = data[-n_steps:]
x_input = x_input.reshape((1, n_steps, n_features))

print(x_input)
yhat = model.predict(x_input, verbose=1)

# print the prediction
print(yhat)