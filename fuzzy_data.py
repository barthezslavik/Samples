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

print(history("Beveren", '2010-01-01'))
print(history("Bergen", '2010-01-01'))
print(hh("Beveren", "Bergen", '2010-01-01'))
print(hh("Bergen", "Beveren", '2010-01-01'))