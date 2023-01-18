import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import process as p

country = 'spain'
period = 9
year = 2009

def get_odds(df):
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
    years = [str(year)[2:] for year in years]
    # for each year, get odds from csv
    for year in years:
        data = pd.read_csv(f'data/{country}/{year}.csv')
        # Get only rows where team1 or team2 is in df
        data = data[data['HomeTeam'].isin(df['team1']) | data['AwayTeam'].isin(df['team2'])]
        # Merge df with data
        merge_df = pd.merge(df, data, how='left', left_on=['date', 'team1', 'team2'], right_on=['Date', 'HomeTeam', 'AwayTeam'])
        # Append to new_df
        new_df = pd.concat([merge_df])

    return new_df

# Neural Network model
X_train, X_test, y_train, y_test, dataset = p.data(country, period, year)
nn_model = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_model.fit(X_train, y_train)
y_pred = nn_model.predict(X_test)
y_pred = np.round(y_pred).astype(int)
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
# Drop all rows where y_pred not equal to 3
df = df[df['y_pred'] == 3]
# Add column called correct and set to 1 if y_test == y_pred
df['correct'] = np.where(df['y_test'] == df['y_pred'], 1, 0)
# Prepend df with date, team1, team2 from corresponding rows from original dataset
df = pd.concat([dataset.loc[df.index][['date', 'team1', 'team2']], df], axis=1)
new_df = get_odds(df)
new_df.to_csv('data/nn.csv', index=False)

# # XGBoost model
# X_train, X_test, y_train, y_test, test_data = p.data(country, period, year)
# xgb_model = xgb.XGBClassifier()
# xgb_model.fit(X_train, y_train)
# y_pred = xgb_model.predict(X_test)