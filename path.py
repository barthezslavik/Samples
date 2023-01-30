import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create new dataset
global_dataset = pd.DataFrame(columns=['Date', 'Team', 'Tour', 'Points'])
d = pd.DataFrame(columns=['Date', 'Team', 'Tour', 'Points'])

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

def get_position(dataset, team, tour):
    data = dataset[dataset['Tour'] == tour]
    data = data.sort_values(by=['Points'], ascending=False)
    # Teams to array
    teams = data['Team'].values
    # Get the index of the team
    index = np.where(teams == team)[0][0]
    return index

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
            if date == '12' or date == '2012':
                print("Processing file: ", file, " - ", first_row['Div'])
                if first_row['Div'] == 'E0':
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
                            # Create a new row
                            new_row = pd.DataFrame([[date, team, tour, points]], columns=['Date', 'Team', 'Tour', 'Points'])
                            # Append the row to the dataset
                            d = d.append(new_row, ignore_index=True)

# Set position
d['Position'] = d.apply(lambda row: get_position(d, row['Team'], row['Tour']), axis=1)

# print(d.tail(50))

# # Plot the for each team the position over all the tours
# for team in teams2(d):
#     data = d[d['Team'] == team]
#     plt.plot(data['Tour'], data['Position'], label=team)
# plt.legend()
# plt.show()

# Remove all except the tour, poins and position
d = d.drop(['Date', 'Team'], axis=1)

# Add column for the next position (the position of the next tour)
d['Next Position'] = d.groupby(['Tour'])['Position'].shift(-1)

# Add column Changed = 1 if the Next Position > Position; 0 otherwise
d['Changed'] = d.apply(lambda row: 1 if row['Next Position'] > row['Position'] else 0, axis=1)

# Drop first 5 tours
d = d[d['Tour'] > 6]

# Drop all where the next position is NaN
d = d.dropna()

# Save the dataset
d.to_csv('data/good/position.csv', index=False)