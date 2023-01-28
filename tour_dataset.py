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

print(df_points.head(50))