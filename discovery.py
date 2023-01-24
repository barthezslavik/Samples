import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = []

# Walk in the directory and get all the files
for root, dirs, files in os.walk("data/discovery"):
    for file in files:
        # print("Processing file: ", file)
        if file.endswith(".csv"):
            try:
                # open the file as dataframe
                df = pd.read_csv(os.path.join(root, file))
                # Add column bet = 0
                df["bet"] = 0
                # add column bet equal value from BWH column if FTHG > FTAG else -1
                if "BWH" in df.columns:
                    df["bet"] = df.apply(lambda x: x["BWH"] - 1 if x["FTHG"] > x["FTAG"] else -1, axis=1)
                    # df["bet"] = df.apply(lambda x: x["BWD"] - 1 if x["FTHG"] == x["FTAG"] else -1, axis=1)
                    # df["bet"] = df.apply(lambda x: x["BWA"] - 1 if x["FTHG"] < x["FTAG"] else -1, axis=1)
                elif "B365D" in df.columns:
                    df["bet"] = df.apply(lambda x: x["B365H"] - 1 if x["FTHG"] > x["FTAG"] else -1, axis=1)
                    # df["bet"] = df.apply(lambda x: x["B365D"] - 1 if x["FTHG"] == x["FTAG"] else -1, axis=1)
                    # df["bet"] = df.apply(lambda x: x["B365A"] - 1 if x["FTHG"] < x["FTAG"] else -1, axis=1)
                # print("Sum for file: ", file, " is: ", sum)
                # Get first row of the dataframe
                first_row = df.iloc[1]
                if os.path.join(root, file) == 'data/discovery/data (13)/E0.csv':
                    print(first_row['Date'], first_row['Div'], sum, os.path.join(root, file))
                    # Sum the bet column
                    sum = df["bet"].sum()
                    print("Sum for file: ", file, " is: ", sum)
                    # print(df.head(50))
                    # Save df to csv
                    # df.to_csv("data/discovery_e0.csv", index=False)
                # Push to data list
                # data.append([first_row['Date'], first_row['Div'], sum, os.path.join(root, file)])
            except:
               print("Error in file: ", os.path.join(root, file))
               pass

# Sort the data by sum
data = sorted(data, key=lambda x: x[2], reverse=True)

# Drop zero values
data = [row for row in data if row[2] != 0]

# Get the Div column
div = [row[1] for row in data]
# Get the Sum column
sum = [row[2] for row in data]

# Plot the data
plt.scatter(div, sum)

# plt.show()