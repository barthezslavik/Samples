import os
from multiprocessing import Pool
import pandas as pd

iteration = 0;
limit = 1000;
dataset = pd.read_csv("data/global.csv")
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
    dataset = dataset[dataset["HomeTeam"] == team]
    # Filter the lines for the date
    dataset = dataset[dataset["Date"] <= date]
    # Remove all expect the result
    dataset = dataset["Y"]
    # Return last 5 results
    return dataset[-5:]

# Head to head for two teams for a specific date
def hh(dataset, team1, team2, date):
    # Filter the lines for the team1
    dataset = dataset[dataset["HomeTeam"] == team1]
    # Filter the lines for the team2
    dataset = dataset[dataset["AwayTeam"] == team2]
    # Filter the lines for the date
    dataset = dataset[dataset["Date"] <= date]
    # Remove all expect the result
    dataset = dataset["Y"]
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
        # Print the iteration
        print("Iteration: " + str(index) + '/' + str(len(dataset)))
        # Get the history for the home team
        home_history = history(dataset, row["HomeTeam"], row["Date"])
        # Get the history for the away team
        away_history = history(dataset, row["AwayTeam"], row["Date"])
        # Get the head to head for the two teams
        hh_history = hh(dataset, row["HomeTeam"], row["AwayTeam"], row["Date"])
        # Get the head to head for the two teams
        hh_history_away = hh(dataset, row["AwayTeam"], row["HomeTeam"], row["Date"])

        # Add the history for the home team
        if len(home_history) > 0:
            dataset.at[index, "x1"] = home_history.iloc[-1]
        if len(home_history) > 1:
            dataset.at[index, "x2"] = home_history.iloc[-2]
        if len(home_history) > 2:
            dataset.at[index, "x3"] = home_history.iloc[-3]
        if len(home_history) > 3:
            dataset.at[index, "x4"] = home_history.iloc[-4]
        if len(home_history) > 4:
            dataset.at[index, "x5"] = home_history.iloc[-5]

        # Add the history for the away team
        if len(away_history) > 0:
            dataset.at[index, "x6"] = away_history.iloc[-1]
        if len(away_history) > 1:
            dataset.at[index, "x7"] = away_history.iloc[-2]
        if len(away_history) > 2:
            dataset.at[index, "x8"] = away_history.iloc[-3]
        if len(away_history) > 3:
            dataset.at[index, "x9"] = away_history.iloc[-4]
        if len(away_history) > 4:
            dataset.at[index, "x10"] = away_history.iloc[-5]

        # Add the head to head for the two teams
        dataset.at[index, "x11"] = hh_history.iloc[-1]

        if len(hh_history_away) > 0:
            dataset.at[index, "x12"] = hh_history_away.iloc[-1]

        # Drop row if x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, Y is null
        if pd.isnull(dataset.at[index, "x1"]) or pd.isnull(dataset.at[index, "x2"]) or pd.isnull(dataset.at[index, "x3"]) or pd.isnull(dataset.at[index, "x4"]) or pd.isnull(dataset.at[index, "x5"]) or pd.isnull(dataset.at[index, "x6"]) or pd.isnull(dataset.at[index, "x7"]) or pd.isnull(dataset.at[index, "x8"]) or pd.isnull(dataset.at[index, "x9"]) or pd.isnull(dataset.at[index, "x10"]) or pd.isnull(dataset.at[index, "x11"]) or pd.isnull(dataset.at[index, "x12"]) or pd.isnull(dataset.at[index, "Y"]):
            dataset = dataset.drop(index)


    # Drop FTR
    dataset = dataset.drop("FTR", axis=1)

    # Save the dataset
    dataset.to_csv("data/global_train.csv", index=False)

build_dataset(dataset)