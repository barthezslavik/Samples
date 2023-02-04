import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

global_data = []

def get_all_matches(dataset, team):
    return dataset[(dataset['HomeTeam'] == team) | (dataset['AwayTeam'] == team)]

def teams(dataset):
    return dataset['HomeTeam'].unique()

def teams2(dataset):
    return dataset['Team'].unique()

def get_points(dataset, team, date):
    matches = get_all_matches(dataset, team)
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
    if diff < -2:
        return 'BL'
    elif diff >= -2 and diff < -1:
        return 'SL'
    elif diff == 0:
        return 'D'
    elif diff > 1 and diff <= 2:
        return 'SW'
    elif diff > 2:
        return 'BW'

# Get last 5 matches
def get_last_5(dataset, team, date):
    matches = get_all_matches(dataset, team)
    matches = matches[matches['Date'] < date]
    matches = matches.tail(5)
    if (len(matches)) > 4:
        print("Last 5 matches for ", team, " at ", date)
        print(matches)
        exit()
    return matches

# Get last 2 matches head to head
def get_last_2_head_to_head(dataset, team1, team2, date):
    matches = dataset[(dataset['HomeTeam'] == team1) & (dataset['AwayTeam'] == team2)]
    matches = matches[matches['Date'] < date]
    matches = matches.tail(2)
    return matches

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
            if date == '13' or date == '2013':
                print("Processing file: ", file, " - ", first_row['Div'])
                if first_row['Div'] == 'E0':
                    # Create new dataset
                    d = pd.DataFrame(columns=['Date', 'Team', 'Tour', 'Points'])
                    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
                    # For each team
                    for team in teams(df):
                        # Get all matches for that team
                        matches = get_all_matches(df, team)
                        # For each match
                        for i in range(len(matches)):
                            # Get the date of the match
                            date = matches.iloc[i]['Date']
                            # Get the position of the team at that date
                            tour = len(matches[matches['Date'] < date])
                            # Points of the team
                            points = get_points(matches, team, date)
                            # Last 5 matches
                            last_5 = get_last_5(matches, team, date)
                            # Create a new row
                            new_row = pd.DataFrame([[date, team, tour, points]], columns=['Date', 'Team', 'Tour', 'Points'])
                            # Append the row to the dataset
                            d = d.append(new_row, ignore_index=True)

                    # Remove all except the tour, poins and position
                    d = d.drop(['Date', 'Team'], axis=1)

                    # Drop first 5 tours
                    d = d[d['Tour'] > 6]

                    # Drop all where the next position is NaN
                    d = d.dropna()

                    global_data.append(d)

# Concatenate all the datasets
global_data = pd.concat(global_data)

# Add column Changed = 2 if the Next Position > Position; 1 if the Next Position == Position; 0 otherwise
# global_data['Changed'] = global_data.apply(lambda row: 2 if row['Next Position'] > row['Position'] else 1 if row['Next Position'] == row['Position'] else 0, axis=1)

# Save the dataset
global_data.to_csv('data/good/fuzzy.csv', index=False)