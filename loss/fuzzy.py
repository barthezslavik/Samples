import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

global_data = []

def get_matches(dataset, team):
    return dataset[(dataset['HomeTeam'] == team) | (dataset['AwayTeam'] == team)]

def get_all_matches(dataset, home, away):
    return dataset[(dataset['HomeTeam'] == home) | (dataset['AwayTeam'] == away) | (dataset['HomeTeam'] == away) | (dataset['AwayTeam'] == home)]

def get_points(dataset, team, date):
    matches = get_matches(dataset, team)
    points = 0
    for i in range(len(matches)):
        if matches.iloc[i]['Date'] < date:
            if matches.iloc[i]['HomeTeam'] == team:
                if matches.iloc[i]['FTHG'] > matches.iloc[i]['FTAG']:
                    points += 3
                elif matches.iloc[i]['FTHG'] == matches.iloc[i]['FTAG']:
                    points += 1
            elif matches.iloc[i]['AwayTeam'] == team:
                if matches.iloc[i]['FTHG'] < matches.iloc[i]['FTAG']:
                    points += 3
                elif matches.iloc[i]['FTHG'] == matches.iloc[i]['FTAG']:
                    points += 1
    return points

# Reverse outcome name
def reverse_outcome_name(outcome):
    if outcome == 'BW':
        return 'BL'
    elif outcome == 'SW':
        return 'SL'
    elif outcome == 'D':
        return 'D'
    elif outcome == 'SL':
        return 'SW'
    elif outcome == 'BL':
        return 'BW'

# Get outcome name
def get_outcome_name(diff):
    if diff <= -2:
        return 'BL'
    elif diff == -2 or diff == -1:
        return 'SL'
    elif diff == 0:
        return 'D'
    elif diff == 1 or diff == 2:
        return 'SW'
    elif diff >= 2:
        return 'BW'

# Get last 5 matches
def get_last_5(dataset, team, date):
    matches = get_matches(dataset, team)
    matches = matches[matches['Date'] < date]
    matches = matches.tail(5)
    names = []
    for match in matches.iterrows():
        if match[1]['HomeTeam'] == team:
            names.append(get_outcome_name(match[1]['FTHG'] - match[1]['FTAG']))
        elif match[1]['AwayTeam'] == team:
            names.append(get_outcome_name(match[1]['FTAG'] - match[1]['FTHG']))

    return names

# Get last 2 matches head to head
def get_last_2_head_to_head(dataset, team1, team2, date):
    matches = dataset[(dataset['HomeTeam'] == team1) & (dataset['AwayTeam'] == team2) | (dataset['HomeTeam'] == team2) & (dataset['AwayTeam'] == team1)]
    matches = matches[matches['Date'] < date]
    matches = matches.tail(2)
    names = []
    for match in matches.iterrows():
        if match[1]['HomeTeam'] == team1:
            names.append(get_outcome_name(match[1]['FTHG'] - match[1]['FTAG']))
        elif match[1]['AwayTeam'] == team1:
            names.append(get_outcome_name(match[1]['FTAG'] - match[1]['FTHG']))
    return names

dfs = []

# Walk in the directory and get all the files
for root, dirs, files in os.walk("data/discovery"):
    for file in files:
        # print("Processing file: ", file)
        if file.endswith(".csv"):
            # open the file as dataframe
            df = pd.read_csv(os.path.join(root, file))
            # Get first row of the dataframe
            first_row = df.iloc[1]
            date = first_row['Date'].split('/')[2]
            years = ['13', '2013', '14', '2014']
            if date in years:
                if first_row['Div'] == 'E0':
                    print(os.path.join(root, file))
                    # Merge the dataframes
                    dfs.append(df)

# Merge all the dataframes
dfs = pd.concat(dfs)

# Create new dataset
d = pd.DataFrame(columns=['Date', 'Home', 'Away', 'Tour', 'Points', 'H', 'D', 'A', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'Y'])
dfs['Date'] = pd.to_datetime(dfs['Date'], format='%d/%m/%y')
# For each team
for match in dfs.iterrows():
    home = match[1]['HomeTeam']
    away = match[1]['AwayTeam']
    print(match[1]['Date'], home, '-', away)
    # Get the date of the match
    date = match[1]['Date']
    # Points
    home_points = get_points(dfs, home, date)
    away_points = get_points(dfs, away, date)
    # Last 5 matches
    home_last_5 = get_last_5(dfs, home, date)
    away_last_5 = get_last_5(dfs, away, date)
    # Last 2 matches head to head
    head_to_head = get_last_2_head_to_head(dfs, home, away, date)
    # Result
    result = get_outcome_name(match[1]['FTHG'] - match[1]['FTAG'])
    # Odds
    h_odd = match[1]['B365H']
    d_odd = match[1]['B365D']
    a_odd = match[1]['B365A']
    # Create new row
    if (len(home_last_5) == 5) and (len(away_last_5) == 5) and (len(head_to_head) == 2):
        new_row = pd.DataFrame([[date, home, away, home_points, h_odd, d_odd, a_odd,
                                home_last_5[0], home_last_5[1], home_last_5[2], home_last_5[3], home_last_5[4], 
                                away_last_5[0], away_last_5[1], away_last_5[2], away_last_5[3], away_last_5[4], 
                                head_to_head[0], head_to_head[1], result]], 
                                columns=['Date', 'Home', 'Away', 'Points', 'H', 'D', 'A', 'X1', 'X2', 'X3', 'X4', 'X5',
                                        'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'Y'])
        # Append the row to the dataset
        global_data.append(new_row)

# Merge all the dataframes
global_data = pd.concat(global_data)

# Save the dataset
global_data.to_csv('data/good/fuzzy.csv', index=False)