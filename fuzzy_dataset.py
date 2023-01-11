import os

# Walk in the directory and get all the files
# for root, dirs, files in os.walk("data/fuzzy"):
#     for file in files:
#         if file.endswith(".csv"):
#             # open the file
#             with open(os.path.join(root, file)) as f:
#                 # read the second line
#                 line = f.readlines()[1]
#                 # read the second column
#                 line = line.split(",")[1]
#                 # get the month and year and remove the day
#                 year =line.split(" ")[0].split("/")[2]
#                 # rename the file to year.csv
#                 os.rename(os.path.join(root, file), os.path.join(root, year + ".csv"))

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
        if file.endswith(".csv"):
            # open the file
            with open(os.path.join(root, file)) as f:
                # Get all matches for each team
                matches = f.readlines()[1:]
                # Create a dictionary to store the matches
                teams = {}
                # Loop through all the matches
                for match in matches:
                    # Get the teams
                    team1 = match.split(",")[2]
                    team2 = match.split(",")[3]
                    # Check if the team is in the dictionary
                    if team1 not in teams:
                        # If not add it
                        teams[team1] = []
                    # Check if the team is in the dictionary
                    if team2 not in teams:
                        # If not add it
                        teams[team2] = []
                    # Add the match to the dictionary
                    #teams[team1].append(match)
                    #teams[team2].append(match)
                    teams[team1].append(get_result_name(int(match.split(",")[4]), int(match.split(",")[5])))
                    teams[team2].append(get_result_name(int(match.split(",")[5]), int(match.split(",")[4])))
                print(teams)