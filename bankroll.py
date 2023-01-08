import random

# Initialize bankroll
bankroll = 100

# Set minimum and maximum bet sizes
min_bet = 1
max_bet = 10

# Set the number of rounds to play
rounds = 1000

# Play the rounds
for i in range(rounds):
    # Select a random percentage of the bankroll to bet
    bet_percentage = random.uniform(0, 0.5)
    bet_size = int(bankroll * bet_percentage)
    
    # Ensure the bet size is within the allowed range
    bet_size = max(min_bet, min(bet_size, max_bet))
    
    # Determine the outcome of the bet (win or lose)
    outcome = random.choice([1, -1])
    
    # Update the bankroll based on the outcome of the bet
    bankroll += outcome * bet_size
    
# Print the final bankroll
print(bankroll)
