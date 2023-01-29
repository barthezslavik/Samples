import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = []

# # Create new dataset
# d = pd.DataFrame(columns=['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'H', 'D', 'A'])

# # Walk in the directory and get all the files
# for root, dirs, files in os.walk("data/discovery"):
#     for file in files:
#         # print("Processing file: ", file)
#         if file.endswith(".csv"):
#             # open the file as dataframe
#             df = pd.read_csv(os.path.join(root, file))
#             # Get first row of the dataframe
#             first_row = df.iloc[1]
#             date = first_row['Date'].split('/')[2]
#             # if date == '16' or date == '2016':
#             if date == '14' or date == '2014':
#                 if "BWH" in df.columns:
#                     df["H"] = df["BWH"]
#                     df["D"] = df["BWD"]
#                     df["A"] = df["BWA"]
#                 elif "B365H" in df.columns:
#                     df["H"] = df["B365H"]
#                     df["D"] = df["B365D"]
#                     df["A"] = df["B365A"]
#                 elif "IWH" in df.columns:
#                     df["H"] = df["IWH"]
#                     df["D"] = df["IWD"]
#                     df["A"] = df["IWA"]
#                 elif "LBH" in df.columns:
#                     df["H"] = df["LBH"]
#                     df["D"] = df["LBD"]
#                     df["A"] = df["LBA"]
#                 elif "WHH" in df.columns:
#                     df["H"] = df["WHH"]
#                     df["D"] = df["WHD"]
#                     df["A"] = df["WHA"]
            
#                 # Remove all columns except Date, Div, HomeTeam, AwayTeam, FTHG, FTAG, H, D, A
#                 df = df[['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'H', 'D', 'A']]
                
#                 # Append the dataframe to the new dataset
#                 d = d.append(df, ignore_index=True)

# # Save the new dataset
# d.to_csv("data/good/discovery14.csv", index=False)
d = pd.read_csv("data/good/discovery14.csv")

def name(diff):
    if diff <= -3:
        return "BL"
    elif diff == -2 or diff == -1:
        return "SL"
    elif diff == 0:
        return "D"
    elif diff == 1 or diff == 2:
        return "SW"
    elif diff >= 3:
        return "BW"

def history(dataset, team, date):
    # Filter the lines for the team
    dataset = dataset[dataset["HomeTeam"] == team]
    # Filter the lines for the date
    dataset = dataset[dataset["Date"] < date]
    # Remove all expect the result
    dataset = dataset["FTHG"] - dataset["FTAG"]
    # Apply the name function
    dataset = dataset.apply(name)
    # Return last 5 results
    return dataset[-5:]

def hh(dataset, team1, team2, date):
    # Filter the lines for the team1
    dataset = dataset[dataset["HomeTeam"] == team1]
    # Filter the lines for the team2
    dataset = dataset[dataset["AwayTeam"] == team2]
    # Filter the lines for the date
    dataset = dataset[dataset["Date"] < date]
    # Remove all expect the result
    dataset = dataset["FTHG"] - dataset["FTAG"]
    # Apply the name function
    dataset = dataset.apply(name)
    # Return last 2 results
    return dataset[-2:]

data = []
# For each macth create x1, x2, x3, x4, x5: last 5 matches of the home team
# x6, x7, x8, x9, x10: last 5 matches of the away team
# x11, x12, last 2 matches head to head of the home team and the away team
for index, row in d.iterrows():
    # if index < 151 or index >= 152:
        # continue
    # print("Processing match: ", row)
    # Get the date
    date = row['Date']
    div = row['Div']
    # Get the home team
    home_team = row['HomeTeam']
    # Get the away team
    away_team = row['AwayTeam']
    # Get the result
    result = row['FTHG'] - row['FTAG']
    # Apply the name function
    result = name(result)
    # Get the odds
    odds = [row['H'], row['D'], row['A']]
    
    # Get the last 5 matches of the home team
    home_team_last_5 = history(d, home_team, date)
    # print(home_team_last_5)
    # Get the last 5 matches of the away team
    away_team_last_5 = history(d, away_team, date)
    # Get the last 2 matches head to head of the home team and the away team
    head_to_head_last_2 = hh(d, home_team, away_team, date)
    
    # Create the new row
    new_row = [date, div, home_team, away_team, result, odds[0], odds[1], odds[2]]
    new_row.extend(home_team_last_5)
    new_row.extend(away_team_last_5)
    new_row.extend(head_to_head_last_2)
    # print(len(new_row))
    print(new_row)
    
    # Append the new row to the dataset
    data.append(new_row)