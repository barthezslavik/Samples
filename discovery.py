import os
import pandas as pd

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
                    df["bet"] = df.apply(lambda x: x["BWH"] if x["FTHG"] > x["FTAG"] else -1, axis=1)
                elif "B365" in df.columns:
                    df["bet"] = df.apply(lambda x: x["B365"] if x["FTHG"] > x["FTAG"] else -1, axis=1)
                # Sum the bet column
                sum = df["bet"].sum()
                # print("Sum for file: ", file, " is: ", sum)
                # Get first row of the dataframe
                first_row = df.iloc[1]
                # print(first_row['Date'], first_row['Div'], sum)
                # Push to data list
                data.append([first_row['Date'], first_row['Div'], sum, os.path.join(root, file)])
            except:
               print("Error in file: ", os.path.join(root, file))
               pass

# Sort the data by sum
data = sorted(data, key=lambda x: x[2], reverse=True)

# Remove data with sum <= 0
data = [row for row in data if row[2] > 0]

# Plot distribution between Div and Sum
import matplotlib.pyplot as plt
import numpy as np

# Get the Div column
div = [row[1] for row in data]
# Get the Sum column
sum = [row[2] for row in data]

# Plot the data
plt.scatter(div, sum)
plt.show()