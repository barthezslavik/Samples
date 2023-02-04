import random
import pandas as pd

# Load the timeseries data
data = pd.read_csv('data/crypto_data.csv')
more = 0
less = 0

for j in range(100):
    # Initialize the bankroll and cryptocurrency holdings
    bankroll = 100
    crypto = 0

    # Set the number of rounds to play
    rounds = 20

    # Play the rounds
    for i in range(rounds):
        # Select a random percentage of the bankroll to use for the trade
        trade_percentage = random.uniform(0, 0.5)
        trade_size = bankroll * trade_percentage
        
        # Determine the outcome of the trade (buy or sell)
        outcome = random.choice(['buy', 'sell'])
        
        # Get the cost of the cryptocurrency at the current time
        cost = data.iloc[i]['cost']
        
        if outcome == 'buy':
            # Calculate the number of units of cryptocurrency that can be bought
            units = trade_size / cost
            
            # Update the bankroll and cryptocurrency holdings
            bankroll -= units * cost
            crypto += units
        elif outcome == 'sell':
            # Sell all of the cryptocurrency
            bankroll += crypto * cost
            crypto = 0

    # Sell all of the cryptocurrency at the end of the rounds
    bankroll += crypto * data.iloc[-1]['cost']
    crypto = 0
    
    if bankroll > 100:
        more += 1

    if bankroll < 100:
        less += 1

    # Print the final bankroll and cryptocurrency holdings
    # print(f'Final bankroll: {bankroll}')
    # print(f'Final cryptocurrency holdings: {crypto}')

print(f'more: {more} less: {less}')