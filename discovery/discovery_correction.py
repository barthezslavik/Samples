import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
#             if date == '12' or date == '13' or date == '14' or date == '2012' or date == '2013' or date == '2014':
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

d['Date'] = pd.to_datetime(d['Date'], format='%d/%m/%y')

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
    # Filter the lines for the HomeTeam vs AwayTeam
    dataset = dataset[((dataset["HomeTeam"] == team1) & (dataset["AwayTeam"] == team2)) |
     ((dataset["HomeTeam"] == team2) & (dataset["AwayTeam"] == team1))]
    # Filter the lines for the date
    dataset = dataset[dataset["Date"] < date]
    # Remove all expect the result
    dataset = dataset["FTHG"] - dataset["FTAG"]
    # Apply the name function
    dataset = dataset.apply(name)
    # Return last 2 results
    return dataset[-2:]

def scored(dataset, team, date):
    print(team)
    # Filter the lines for the HomeTeam
    dataset = dataset[(dataset["HomeTeam"] == team) | (dataset["AwayTeam"] == team)]

    # Filter the lines for the date
    dataset = dataset[dataset["Date"] < date]
    if len(dataset) == 0:
        return 0

    # Set the goals to FTHG if the team is the HomeTeam
    dataset.loc[dataset["HomeTeam"] == team, "goals"] = dataset["FTHG"]
    # Set the goals to FTAG if the team is the AwayTeam
    dataset.loc[dataset["AwayTeam"] == team, "goals"] = dataset["FTAG"]
    # Sum the goals from last 5 matches
    return dataset["goals"][-5:].sum()

data = []
# For each macth create x1, x2, x3, x4, x5: last 5 matches of the home team
# x6, x7, x8, x9, x10: last 5 matches of the away team
# x11, x12, last 2 matches head to head of the home team and the away team
for index, row in d.iterrows():
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
    # print(row)
    result = name(result)
    # Get the odds
    odds = [row['H'], row['D'], row['A']]
    
    # Get the last 5 matches of the home team
    home_team_last_5 = history(d, home_team, date)
    # Get the last 5 matches of the away team
    away_team_last_5 = history(d, away_team, date)
    # Get the last 2 matches head to head of the home team and the away team
    head_to_head_last_2 = hh(d, home_team, away_team, date)
    
    # Create the new row
    new_row = [date, div, home_team, away_team, odds[0], odds[1], odds[2], result]
    new_row.extend(home_team_last_5)
    new_row.extend(away_team_last_5)
    new_row.extend(head_to_head_last_2)
    new_row.append(scored(d, home_team, date))
    new_row.append(scored(d, away_team, date))
    print(new_row)
    
    if(len(new_row) == 22):
        # Append the new row to the dataset
        data.append(new_row)

# Create the new dataset
dataset = pd.DataFrame(data, columns=['Date', 'Div', 'HomeTeam', 'AwayTeam', 'H', 'D', 'A', 'Y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12','HS','AS'])
# Save the new dataset
dataset.to_csv("data/good/test.csv", index=False)