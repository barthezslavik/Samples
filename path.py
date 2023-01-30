import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create new dataset
d = pd.DataFrame(columns=['Date', 'Team', 'Tour', 'Position'])

def get_all_matches(dataset, team):
    return dataset[(dataset['HomeTeam'] == team) | (dataset['AwayTeam'] == team)]

def teams(dataset):
    return dataset['HomeTeam'].unique()

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
                            # Position of the team
                            position = 0
                            # Create a new row
                            new_row = pd.DataFrame([[date, team, tour, position]], columns=['Date', 'Team', 'Tour', 'Position'])
                            # Append the row to the dataset
                            d = d.append(new_row, ignore_index=True)

print(d.head(50))