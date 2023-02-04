import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def format_year(year):
    if len(year) == 4:
        year = year[2:]
    return year

# Convert 04/08/95 to 1995/08/04
def format_date(date):
    date = str(date)
    if date != 'nan':
        date = date.split("/")
        if len(date[2]) == 2:
            if int(date[2]) < 20:
                date[2] = "20" + date[2]
            else:
                date[2] = "19" + date[2]
        return "-".join(date)
    else:
        return ""

def get_result_name(home_score, away_score):
    diff = home_score - away_score
    if diff <= -3:
        return "BL"
    elif diff == -2 or diff == -1:
        return "SL"
    elif diff == 0:
        return "D"
    elif diff == 1 or diff == 2:
        return "SW"
    elif diff >= 3:
        return "BW"

data = []
# Create result dataset
result = pd.DataFrame(columns=['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'H', 'D', 'A'])

# Walk in the directory and get all the files
for root, dirs, files in os.walk("data/discovery"):
    for file in files:
        # print("Processing file: ", file)
        if file.endswith(".csv"):
            # open the file as dataframe
            df = pd.read_csv(os.path.join(root, file))
            # add column bet equal value from BWH column if FTHG > FTAG else -1
            if "BWH" in df.columns:
                # Set H column to BWH value
                df["H"] = df["BWH"]
                # Set D column to BWD value
                df["D"] = df["BWD"]
                # Set A column to BWA value
                df["A"] = df["BWA"]
            elif "B365D" in df.columns:
                # Set H column to B365H value
                df["H"] = df["B365H"]
                # Set D column to B365D value
                df["D"] = df["B365D"]
                # Set A column to B365A value
                df["A"] = df["B365A"]
            elif "GBH" in df.columns:
                # Set H column to GBH value
                df["H"] = df["GBH"]
                # Set D column to GBH value
                df["D"] = df["GBD"]
                # Set A column to GBA value
                df["A"] = df["GBA"]
            elif "WHH" in df.columns:
                # Set H column to WHH value
                df["H"] = df["WHH"]
                # Set D column to WHD value
                df["D"] = df["WHD"]
                # Set A column to WHA value
                df["A"] = df["WHA"]

            if "HT" in df.columns:
                df["HomeTeam"] = df["HT"]
                df["AwayTeam"] = df["AT"]

            # Format date
            df["Date"] = df["Date"].apply(format_date)
            # Add column y = get_result_name(FTHG, FTAG)
            df["Y"] = df.apply(lambda row: get_result_name(row["FTHG"], row["FTAG"]), axis=1)
            # Drop all columns except Date, Div, HomeTeam, AwayTeam, FTAG, FTHG, H, D, A
            df = df[['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTAG', 'FTHG', 'H', 'D', 'A', 'Y']]
            # Append the dataframe to the result
            result = result.append(df, ignore_index=True)

# Save the result to csv file
result.to_csv("data/global.csv", index=False)