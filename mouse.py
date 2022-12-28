# Import the Pygame library and initialize the game engine
import pygame
pygame.init()
done = False

# Define the dimensions of the game screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Create a Surface object to represent the game screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Create a game loop to run until the user quits
while not done:
    # Check for events and handle them
    for event in pygame.event.get():
        # If the user closes the window, quit the game
        if event.type == pygame.QUIT:
            done = True

        # If the user clicks the mouse, handle the click
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Get the position of the mouse cursor at the time of the click
            mouse_x, mouse_y = event.pos

            # Perform some action based on the clicked position
            print(event.pos)

    # Update the game screen and wait for the next frame
    pygame.display.flip()