import os

# Walk in the directory and get all the files
for root, dirs, files in os.walk("data/fuzzy"):
    for file in files:
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