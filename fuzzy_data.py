import os

def invert_result_for_away_team(result):
    result = clear(result)
    if result == "BW":
        return "BL"
    elif result == "SL":
        return "SW"
    elif result == "SW":
        return "SL"
    elif result == "BL":
        return "BW"
    elif result == "D":
        return "D"

def clear(result):
    return result.replace("\n", "")

# History for specific team
def history(team, date):
    print(team, date)
    # Read the file as lines
    with open("data/fuzzy/fuzzy2.csv", "r") as f:
        lines = f.readlines()
    # Split each line into columns
    lines = [line.split(",") for line in lines]
    # Remove the last column for each line
    lines = [line[0:6] for line in lines]
    # Filter the lines for the team
    lines = [line for line in lines if line[1] == team or line[2] == team]
    # Filter the lines for the date
    lines = [line for line in lines if line[0] <= date]
    # Invert the result for the away team
    lines = [[line[0], line[1], line[2], line[3], line[4], invert_result_for_away_team(line[5])] if line[2] == team else line for line in lines]
    # Remove all expect the result
    lines = [clear(line[5]) for line in lines]
    print(lines[-5:])
    return lines[-5:]

# Head to head for two teams for a specific date
def hh(team1, team2, date):
    # Read the file as lines
    with open("data/fuzzy/fuzzy2.csv", "r") as f:
        lines = f.readlines()
    # Split each line into columns
    lines = [line.split(",") for line in lines]
    # Remove the last column for each line
    lines = [line[0:6] for line in lines]
    # Filter the lines for the team
    lines = [line for line in lines if (line[1] == team1 and line[2] == team2) or (line[1] == team2 and line[2] == team1)]
    # Filter the lines for the date
    lines = [line for line in lines if line[0] <= date]
    # Invert the result for the away team
    lines = [[line[0], line[1], line[2], line[3], line[4], invert_result_for_away_team(line[5])] if line[2] == team1 else line for line in lines]
    # Remove all expect the result
    lines = [clear(line[5]) for line in lines]
    # Return last 2 results
    return lines[-2:]

# print(history("Beveren", '2010-01-01'))
# print(history("Bergen", '2010-01-01'))
# print(hh("Beveren", "Bergen", '2010-01-01'))
# print(hh("Bergen", "Beveren", '2010-01-01'))

# Build dataset with columns: x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, y
# x1, x2, x3, x4, x5: result of the last 5 games for the home team
# x6, x7, x8, x9, x10: result of the last 5 games for the away team
# x11, x12: result of the last 2 games between the two teams
# y: result of the game (BW, SL, SW, BL, D)
def build_dataset():
    # Read the file as lines
    with open("data/fuzzy/fuzzy2.csv", "r") as f:
        lines = f.readlines()
    # Split each line into columns
    lines = [line.split(",") for line in lines]
    # Remove the last column for each line
    lines = [line[0:6] for line in lines]
    # Remove the header
    lines = lines[1:]
    # Invert the result for the away team
    lines = [[line[0], line[1], line[2], line[3], line[4], invert_result_for_away_team(line[5])] if line[2] == line[3] else line for line in lines]
    # Remove all expect the result
    lines = [[line[0], line[1], line[2], line[3], line[4], clear(line[5])] for line in lines]
    # Build the dataset
    dataset = []
    # Get 10 first lines
    lines = lines[0:1000]
    for line in lines:
        # Get the history for the home team
        home_team_history = history(line[1], line[0])
        # Get the history for the away team
        away_team_history = history(line[2], line[0])
        # Get the head to head for the two teams
        hhh = hh(line[1], line[2], line[0])
        # Append the dataset
        dataset.append(home_team_history + away_team_history + hhh + [line[5]])
    return dataset

dataset = build_dataset()
# Write the dataset to a file
with open("data/fuzzy/fuzzy3.csv", "w") as f:
    for line in dataset:
        f.write(",".join(line) + "\n")