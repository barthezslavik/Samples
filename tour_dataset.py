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
df = df.drop(["FTHG", "FTAG"], axis=1)

# Create a new dataframe for each tour
tours = df["Tour"].unique()
dfs = {}
for tour in tours:
    dfs[tour] = df[df["Tour"] == tour]
    dfs[tour] = dfs[tour].drop(["Tour"], axis=1)

print(df.head())

# Create a copy of the original dataframe
df_standings = df.copy()

# Group the data by tour
grouped = df_standings.groupby('Tour')

# Calculate total points for each team
df_standings['TotalPoints'] = df_standings['HomePoints'] + df_standings['AwayPoints']

df_standings['Standing'] = None

# Create a new column for team standing
for name, group in grouped:
    group = group.sort_values(by=['TotalPoints'], ascending=False)
    group['Standing'] = range(1, len(group) + 1)
    df_standings.loc[group.index] = group

# pivot the data
df_standings_pivot = df_standings.pivot_table(values=['TotalPoints','Standing'], index=['HomeTeam','Tour'],aggfunc='first')
df_standings_pivot.reset_index(inplace=True)
df_standings_pivot.rename(columns={'HomeTeam':'TeamName'},inplace=True)

df_standings_pivot_away = df_standings.pivot_table(values=['TotalPoints','Standing'], index=['AwayTeam','Tour'],aggfunc='first')
df_standings_pivot_away.reset_index(inplace=True)
df_standings_pivot_away.rename(columns={'AwayTeam':'TeamName'},inplace=True)

# concatenate the result from home and away
df_standings_all = pd.concat([df_standings_pivot,df_standings_pivot_away],ignore_index=True)

print(df_standings_all.head(50))