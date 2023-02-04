import csv
import random
import math

# Load data from CSV file
with open('data/crypto_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Initialize variables
bankroll = 100  # initial bankroll
crypto = 0  # initial amount of cryptocurrency
prev_bankroll = bankroll+1  # store initial bankroll for comparison
prev_crypto = crypto  # store initial cryptocurrency for comparison

# Iterate over data
for i, row in enumerate(data):
    # Get current cost of cryptocurrency
    cost = float(row['cost'])

    # Calculate the entropy of the change in bankroll and cryptocurrency holdings
    S = 0
    if bankroll - prev_bankroll != 0:
        S -= (bankroll - prev_bankroll)*math.log(bankroll - prev_bankroll)
    if crypto - prev_crypto != 0:
        S -= (crypto - prev_crypto)*math.log(crypto - prev_crypto)

    print(f'Entropy: {S}')
    # Choose the action that maximizes the entropy
    if S > 0:
        # Buy cryptocurrency if possible
        if bankroll >= cost:
            bankroll -= cost
            crypto += 1
    else:
        # Sell cryptocurrency if possible
        if crypto > 0:
            bankroll += cost
            crypto -= 1

    # Store current values for comparison in next iteration
    prev_bankroll = bankroll
    prev_crypto = crypto

print(f'Final bankroll: {bankroll}')
print(f'Final cryptocurrency: {crypto}')
