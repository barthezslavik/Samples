import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = []
# Create dataframe with columns Div, Date, Team1, Team2, Win
total = pd.DataFrame(columns=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'Bet'])

# Walk in the directory and get all the files
for root, dirs, files in os.walk("data/current"):
    for file in files:
        # print("Processing file: ", file)
        if file.endswith(".csv"):
            # open the file as dataframe
            df = pd.read_csv(os.path.join(root, file))
            # Add column bet = 0
            df["Bet"] = 0
            # add column bet equal value from BWH column if FTHG > FTAG else -1
            if "BWH" in df.columns:
                df["Bet"] = df.apply(lambda x: x["BWH"] if x["FTHG"] > x["FTAG"] else -1, axis=1)
            elif "B365D" in df.columns:
                df["Bet"] = df.apply(lambda x: x["B365H"] if x["FTHG"] > x["FTAG"] else -1, axis=1)
            # Get all row where Div == 'E1', 'E2', 'E3', 'SP2', 'E0', 'F0', 'SP1', 'F2','D1','T1','N1','D2','B1'
            df = df[df['Div'].isin(['E1', 'E2', 'E3', 'SP2', 'E0', 'F0', 'SP1', 'F2','D1','T1','N1','D2','B1'])]
            # Merge the total dataframe with the new dataframe if df is not empty
            if not df.empty:
                total = pd.concat([total, df], ignore_index=True)

print(total.head(50))
# Sum the bet column
sum = total["Bet"].sum()
print("Sum", sum)

# Save the dataframe to csv
total.to_csv("data/current.csv", index=False)