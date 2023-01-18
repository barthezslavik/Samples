import numpy as np
import pandas as pd
import os
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import process as p

country = 'england'
period = 15
year = 2006

def format_date(date):
    date = date.replace("-", "/")
    date = date.split("/")
    if len(date[2]) == 2:
        if int(date[2]) < 20:
            date[2] = "20" + date[2]
        else:
            date[2] = "19" + date[2]
    return "-".join(date)

def create_odds(df):
    # Init empty dataframe
    new_df = pd.DataFrame()
    # Get years from date column
    years = df['date'].tolist()
    # Get year from each date
    years = [year.split('-')[-1] for year in years]
    # Unique years
    years = list(set(years))
    # COnvert to int
    years = [int(year) for year in years]
    # Sort
    years.sort()
    # Remove '20' from each year
    years = [int(str(year)[2:]) for year in years]
    # Prepend year[0] - 1
    years.insert(0, years[0] - 1)
    # Merge datasets for each year
    for year in years:
        # Read csv
        df = pd.read_csv(f'data/{country}/{year}.csv')
        # Append to new_df
        new_df = new_df.append(df)
        # Change date format
        new_df['Date'] = new_df['Date'].apply(format_date)
        # Save to csv
        new_df.to_csv(f'data/odds.csv', index=False)

X_train, X_test, y_train, y_test, dataset = p.data(country, period, year)

# Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_model.fit(X_train, y_train)
y_pred = nn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

# XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred2 = xgb_model.predict(X_test)
y_pred2 = np.round(y_pred2).astype(int)

# Merge y_pred and y_pred2, if y_pred == 2, use y_pred2, else use y_pred
y_pred3 = np.where(y_pred == 2, y_pred2, y_pred)
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred3})
# Add column called correct and set to 1 if y_test == y_pred
df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)
# Devide correct by total rows
accuracy = df['correct'].sum() / df.shape[0]
print(f'Accuracy: {accuracy}')

# Prepend df with date, team1, team2 from corresponding rows from original dataset
df = pd.concat([dataset.loc[df.index][['date', 'team1', 'team2']], df], axis=1)
# Create odds file
create_odds(df)
# Read odds file
odds = pd.read_csv('data/odds.csv')
# Drop all except Date, HomeTeam, AwayTeam, B365H, B365D, B365A
odds = odds[['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']]
# Merge df and odds if Date, team1, team2 match
df = pd.merge(df, odds, how='left', left_on=['date', 'team1', 'team2'], right_on=['Date', 'HomeTeam', 'AwayTeam'])
# Drop Date, HomeTeam, AwayTeam
df = df.drop(['Date', 'HomeTeam', 'AwayTeam'], axis=1)
# Add profit column
# Set to B365A if y_pred == 0
# Set to B365A if y_pred == 1
# Set to B365D if y_pred == 2
# Set to B365H if y_pred == 3
# Set to B365H if y_pred == 4
df['profit'] = np.where(df['y_pred'] == 0, df['B365A'], np.where(df['y_pred'] == 1, df['B365A'], np.where(df['y_pred'] == 2, df['B365D'], np.where(df['y_pred'] == 3, df['B365H'], np.where(df['y_pred'] == 4, df['B365H'], -1)))))
# Set to -1 if y_test != y_pred
df['profit'] = np.where(df['y_test'] != df['y_pred'], -1, df['profit'])
# Sum profit column
profit = df['profit'].sum()
print(f'Profit: {profit}')

# Write df file
df.to_csv('data/df.csv', index=False)

# Delete odds file
os.remove('data/odds.csv')