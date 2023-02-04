import pygame
import random

# Initialize Pygame
pygame.init()

# Set the window size and background color
window_size = (600, 600)
screen = pygame.display.set_mode(window_size)
bg_color = (255, 255, 255)

# Set the initial price of the stock and the initial cash balance
price = 100
cash = 1000

# Set the probability of selling or buying the stock
sell_prob = 0.5
buy_prob = 1 - sell_prob

# Set the number of time steps to simulate
num_time_steps = 1000

# Set the time step interval
dt = 1  # seconds

# Set the simulation running flag
running = True

# Run the simulation loop
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the price of the stock based on a random change
    price += random.uniform(-1, 1)

    # Decide whether to sell or buy the stock based on the probabilities
    action = random.uniform(0, 1)
    if action < sell_prob:
        # Sell the stock and update the cash balance
        cash += price
    elif action < buy_prob:
        # Buy the stock and update the cash balance
        cash -= price

    # Draw the current price and cash balance on the screen
    screen.fill(bg_color)
    font = pygame.font.Font(None, 36)
    price_text = font.render("Price: ${:.2f}".format(price), True, (0, 0, 0))
    cash_text = font.render("Cash: ${:.2f}".format(cash), True, (0, 0, 0))
    screen.blit(price_text, (10, 10))
    screen.blit(cash_text, (10, 50))

    # Update the display
    pygame.display.flip()

#Quit Pygame
pygame.quit()