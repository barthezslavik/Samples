import pandas as pd

countries = ['england', 'germany', 'italy', 'spain', 'france']

for year in range(1994, 2020):
    # Init empty dataframe
    new_df = pd.DataFrame()
    for country in countries:
        # year = 1994 -> 94
        if len(str(year)) == 4:
            year = str(year)[2:]
        # Read csv
        df = pd.read_csv(f'data/{country}/{year}.csv')
        #Append to new_df
        new_df = new_df.append(df)
    
    # Save to csv
    new_df.to_csv(f'data/total/{year}.csv', index=False)
