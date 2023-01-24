import os
from multiprocessing import Pool
import pandas as pd

iteration = 0;
limit = 100;
dataset = pd.read_csv("data/global.csv")
# Get first 100 lines
dataset = dataset[:limit]

def invert_result_for_away_team(result):
    result = clear(result)
    if result == "BW":
        return "BL"
    elif result == "SL":
        return "SW"
    elif result == "SW":
        return "SL"
    elif result == "BL":
        return "BW"
    elif result == "D":
        return "D"

def clear(result):
    return result.replace("\n", "")

# History for specific team
def history(dataset, team, date):
    # Filter the lines for the team
    dataset = dataset[(dataset["HomeTeam"] == team) | (dataset["AwayTeam"] == team)]
    # Filter the lines for the date
    dataset = dataset[dataset["Date"] <= date]
    # Invert the result for the away team
    dataset.loc[dataset["AwayTeam"] == team, "FTR"] = dataset.loc[dataset["AwayTeam"] == team, "FTR"].apply(invert_result_for_away_team)
    # Remove all expect the result
    dataset = dataset["FTR"]
    print(dataset.head())
    return dataset[-5:]

# Head to head for two teams for a specific date
def hh(dataset, team1, team2, date):
    # Filter the lines for the team
    dataset = dataset[((dataset["HomeTeam"] == team1) & (dataset["AwayTeam"] == team2)) | ((dataset["HomeTeam"] == team2) & (dataset["AwayTeam"] == team1))]
    # Filter the lines for the date
    dataset = dataset[dataset["Date"] <= date]
    # Invert the result for the away team
    dataset.loc[dataset["AwayTeam"] == team1, "FTR"] = dataset.loc[dataset["AwayTeam"] == team1, "FTR"].apply(invert_result_for_away_team)
    # Remove all expect the result
    dataset = dataset["FTR"]
    # Return last 2 results
    return dataset[-2:]

# Build dataset with columns: x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, y
# x1, x2, x3, x4, x5: result of the last 5 games for the home team
# x6, x7, x8, x9, x10: result of the last 5 games for the away team
# x11, x12: result of the last 2 games between the two teams
# y: result of the game (BW, SL, SW, BL, D)
def build_dataset(dataset):
    # For each line in the dataset
    for index, row in dataset.iterrows():
        # Get the history for the home team
        home_history = history(dataset, row["HomeTeam"], row["Date"])
        # Get the history for the away team
        away_history = history(dataset, row["AwayTeam"], row["Date"])
        # Get the head to head for the two teams
        hh_history = hh(dataset, row["HomeTeam"], row["AwayTeam"], row["Date"])
        # Add the result to the dataset
        dataset.loc[index, "x1"] = home_history.iloc[-1]
        dataset.loc[index, "x2"] = home_history.iloc[-2]
        dataset.loc[index, "x3"] = home_history.iloc[-3]
        dataset.loc[index, "x4"] = home_history.iloc[-4]
        dataset.loc[index, "x5"] = home_history.iloc[-5]
        dataset.loc[index, "x6"] = away_history.iloc[-1]
        dataset.loc[index, "x7"] = away_history.iloc[-2]
        dataset.loc[index, "x8"] = away_history.iloc[-3]
        dataset.loc[index, "x9"] = away_history.iloc[-4]
        dataset.loc[index, "x10"] = away_history.iloc[-5]
        dataset.loc[index, "x11"] = hh_history.iloc[-1]
        dataset.loc[index, "x12"] = hh_history.iloc[-2]
        dataset.loc[index, "y"] = row["FTR"]
    # Save the dataset
    print(dataset.head())
    dataset.to_csv("data/global_dataset.csv", index=False)

build_dataset(dataset)