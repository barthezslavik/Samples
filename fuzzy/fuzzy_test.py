import pandas as pd

# read fuzzy5.csv
df = pd.read_csv("data/fuzzy5.csv")
# add column bet equal 0
df["bet"] = 0

# set column bet equal value from H column if outcome = H and prediction = SW, BW
# set column bet equal value from D column if outcome = D and prediction = D
# set column bet equal value from A column if outcome = A and prediction = SL, BL
# else set column bet equal -1
df["bet"] = df.apply(lambda x: x["H"] - 1 if x["outcome"] == "H" and x["prediction"] in ["SW", "BW"] else -1, axis=1)

# add column score equal FTHG + ":" + FTAG
df["score"] = df["home_score"].astype(int).astype(str) + ":" + df["away_score"].astype(int).astype(str)

# Sort by bet
# df = df.sort_values(by=["bet"], ascending=False)

# Drop all except div = E0
df = df[df["div"] == "E0"]

# Group by year in Date column by splitting the date by "-" and taking the last element
df["year"] = df["date"].apply(lambda x: x.split("-")[-1])

# Drop all except year = 2009
df = df[df["year"] == "2009"]

# Save to csv
df.to_csv("data/fuzzy5_e0.csv", index=False)

# print(df.head(50))

# Calculate the outcome distribution
# outcome = df["outcome"].value_counts()

# print the outcome distribution
# print(outcome)

# sum the bet column
sum = df["bet"].sum()

# print the sum by year
print(sum)

# Group sum by Div and year
sum = df.groupby(["div", "year"])["bet"].sum()

# print the sum by Div and year
print(sum)