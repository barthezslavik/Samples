import pandas as pd

# Load the dataset
df = pd.read_csv('data/E0.csv')

# Extract the goal difference for each game
df['diff'] = df['FTHG'] - df['FTAG']

# Map the goal difference to the appropriate outcome category
outcome_map = {
    (-5, -3): 'BL',
    (-2, -1): 'SL',
    (-0.5, 0.5): 'D',
    (1, 2): 'SW',
    (3, 5): 'BW'
}

def map_outcome(diff):
    for key, value in outcome_map.items():
        if key[0] <= diff <= key[1]:
            return value

df['outcome'] = df['diff'].apply(map_outcome)

# Keep only the relevant columns
df = df[['diff', 'outcome']]

# Save the modified dataset to a new file
df.to_csv('modified_dataset.csv', index=False)
