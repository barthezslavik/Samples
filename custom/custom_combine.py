import pandas as pd

def format_date(date):
    try:
        date = date.replace("-", "/")
        date = date.split("/")
        if len(date[2]) == 2:
            if int(date[2]) < 20:
                date[2] = "20" + date[2]
            else:
                date[2] = "19" + date[2]
        return "-".join(date)
    except:
        return date

# Merge fuzzy4.csv and fuzzy.csv to fuzzy5.csv
# Read fuzzy4.csv as dataframe
fuzzy4 = pd.read_csv("data/fuzzy/fuzzy4.csv")

# Read fuzzy.csv as dataframe
fuzzy = pd.read_csv("data/fuzzy/fuzzy.csv")

print(fuzzy.head())

# # Replace date column in fuzzy.csv with formatted date
fuzzy["date"] = fuzzy["date"].apply(format_date)

# Remove empty rows
fuzzy = fuzzy[fuzzy["date"].notnull()]

# Merge fuzzy4.csv and fuzzy.csv by date, team1 and team2
fuzzy = fuzzy.merge(fuzzy4, on=["date", "team1", "team2"], how="outer")

# Drop if x1 or x2 is empty
fuzzy = fuzzy[fuzzy["x1"].notnull()]

# Remove rows where H, D or A is empty
fuzzy = fuzzy[fuzzy["H"].notnull()]
fuzzy = fuzzy[fuzzy["D"].notnull()]
fuzzy = fuzzy[fuzzy["A"].notnull()]

# # Create fuzzy5.csv using fuzzy.csv
fuzzy.to_csv("data/fuzzy/fuzzy5.csv", index=False)