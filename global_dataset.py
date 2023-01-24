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

# Walk in the directory and get all the files
for root, dirs, files in os.walk("data/fuzzy"):
    for file in files:
        # Skip the files with name fuzzy.csv and fuzzy2.csv
        if file not in ["fuzzy.csv", "fuzzy2.csv", "fuzzy3.csv"]:
            if file.endswith(".csv"):
                # open the file
                with open(os.path.join(root, file)) as f:
                    print("Processing file: ", file)
                    # read the second line
                    line = f.readlines()[1]
                    # read the second column
                    line = line.split(",")[1]
                    # get the month and year and remove the day
                    year =line.split(" ")[0].split("/")[2].replace('"', '')
                    year = format_year(year)
                    # rename the file to year.csv
                    os.rename(os.path.join(root, file), os.path.join(root, year + ".csv"))

order = ["95", "96", "97", "98", "99", "00", "01", 
        "02", "03", "04", "05", "06", "07", "08", 
        "09", "10", "11", "12", "13", "14", "15",
        "16", "17", "18", "19", "20", "21", "22"]

with open("data/fuzzy/fuzzy.csv", "w") as f:
    f.write("")
with open("data/fuzzy/fuzzy2.csv", "w") as f:
    f.write("")
with open("data/fuzzy/fuzzy3.csv", "w") as f:
    f.write("")

# Merge all the files into one
for year in order:
    file_name = "data/fuzzy/" + str(year) + ".csv"
    # Open the file
    if os.path.exists(file_name):
        with open(file_name) as f:
            print("Processing file: " + file_name)
            # read the csv file into a dataframe
            df = pd.read_csv(file_name)
            # drop the "Time" column if it exists
            if "Time" in df.columns:
                print("Dropping Time column")
                df = df.drop("Time", axis=1)
                # save the dataframe to the same file
                df.to_csv(file_name, index=False)
            # Read all the lines
            lines = f.readlines()
            lines = [line for line in lines if "HomeTeam" not in line]
            with open("data/fuzzy/fuzzy.csv", "a") as f2:
                # Write all the lines
                f2.writelines(lines)

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
        if line == ['', '', '', '', '']:
            continue

        # line contains '"' remove it
        line = [l.replace('"', '') for l in line]
        # Format the date
        line[0] = format_date(line[0])
        line.append(get_result_name(int(float(line[3])), int(float(line[4]))))
        f.write(",".join(line) + "\n")