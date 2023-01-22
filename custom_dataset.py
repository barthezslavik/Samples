import os
import pandas as pd

# Remove the 2nd column for each line
def remove_2nd_column(line):
    line = line.split(",")
    line = line[:1] + line[3:]
    return ",".join(line)

def format_year(year):
    if len(year) == 4:
        year = year[2:]
    return year

# Convert 04/08/95 to 1995/08/04
def format_date(date):
    date = date.split("/")
    if len(date[2]) == 2:
        if int(date[2]) < 20:
            date[2] = "20" + date[2]
        else:
            date[2] = "19" + date[2]
    return "-".join(date)

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

# Read the file as lines
with open("data/fuzzy/fuzzy.csv", "r") as f:
    lines = f.readlines()

# Split each line into columns
lines = [line.split(",") for line in lines]
# Remove the last column for each line
lines = [line[1:6] for line in lines]

# Write the lines in the file fuzzy2.csv
with open("data/fuzzy/fuzzy2.csv", "w") as f:
    for line in lines:
        print(line)
        if line == ['', '', '', '', ''] or line == ['""', '""', '""', '""', '""']:
            continue

        # line contains '"' remove it
        line = [l.replace('"', '') for l in line]
        # Format the date
        line[0] = format_date(line[0])
        line.append(get_result_name(int(float(line[3])), int(float(line[4]))))
        f.write(",".join(line) + "\n")