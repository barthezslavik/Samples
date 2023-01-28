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

# Drop all the columns except FTHG, FTAG, HomePoints, AwayPoints, Tour
df = df[["HomeTeam", "AwayTeam", "FTHG", "FTAG", "HomePoints", "AwayPoints", "Tour"]]

# Save the dataset
df.to_csv("data/tour/E0_1.csv", index=False)

# Drop the unnecessary columns
df = df.drop(["HomeTeam", "AwayTeam", "FTHG", "FTAG"], axis=1)

# Create a new dataframe for each tour
tours = df["Tour"].unique()
dfs = {}
for tour in tours:
    dfs[tour] = df[df["Tour"] == tour]
    dfs[tour] = dfs[tour].drop(["Tour"], axis=1)

timestep = 1
for tour in tours:
    # Get the tour dataframe
    df = dfs[tour]

    # Create the input dataset
    X_train = np.zeros((df.shape[0] - timestep, timestep, 2))
    for i in range(timestep, df.shape[0]):
        for j in range(timestep):
            X_train[i-timestep][j] = df.iloc[i-j-1, :].values

    # Create the output dataset
    y_train = df.iloc[timestep:, :].values

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model to the data
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Use the model to predict the values for the next tour
    if tour == tours[-1]:
        break
    next_tour_df = dfs[tour+1]
    X_test = np.zeros((next_tour_df.shape[0], timestep, 2))
    for i in range(timestep, next_tour_df.shape[0]):
        for j in range(timestep):
            X_test[i-timestep][j] = next_tour_df.iloc[i-j-1, :].values
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = 0
    for i in range(predictions.shape[0]):
        if predictions[i][0] > predictions[i][1]:
            accuracy += 1
        elif predictions[i][0] == predictions[i][1]:
            accuracy += 0.5
    accuracy /= predictions.shape[0]
    print("Tour: {}, Accuracy: {}".format(tour+1, accuracy))