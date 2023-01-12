import os

# Walk in the directory and get all the files
for root, dirs, files in os.walk("data/fuzzy"):
    for file in files:
        # Skip the files with name fuzzy.csv and fuzzy2.csv
        if file not in ["fuzzy.csv", "fuzzy2.csv"]:
            if file.endswith(".csv"):
                # open the file
                with open(os.path.join(root, file)) as f:
                    # read the second line
                    line = f.readlines()[1]
                    # read the second column
                    line = line.split(",")[1]
                    # get the month and year and remove the day
                    year =line.split(" ")[0].split("/")[2]
                    # rename the file to year.csv
                    os.rename(os.path.join(root, file), os.path.join(root, year + ".csv"))

order = ["95", "96", "97", "98", "99", "00", "01", 
        "02", "03", "04", "05", "06", "07", "08", 
        "09", "10", "11", "12", "13"]

# Merge all the files into one
for year in order:
    # Open the file
    with open("data/fuzzy/" + str(year) + ".csv") as f:
        # Read all the lines
        lines = f.readlines()
        # Open the file
        # Drop lines if it contains "HomeTeam"
        lines = [line for line in lines if "HomeTeam" not in line]
        with open("data/fuzzy/fuzzy.csv", "a") as f2:
            # Write all the lines
            f2.writelines(lines)

# Read the file as lines
with open("data/fuzzy/fuzzy.csv", "r") as f:
    lines = f.readlines()

# print(len(lines))

# Split each line into columns
lines = [line.split(",") for line in lines]
# Remove the last column for each line
lines = [line[1:6] for line in lines]

# Write the lines in the file fuzzy2.csv
with open("data/fuzzy/fuzzy2.csv", "w") as f:
    for line in lines:
        f.write(",".join(line) + "\n")