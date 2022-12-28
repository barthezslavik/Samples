import pygame
import random

# Initialize Pygame
pygame.init()

# Set the window size
window_size = (400, 200)

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the title of the window
pygame.display.set_caption("Molecule Simulation")

# Define the colors
black = (0, 0, 0)
white = (255, 255, 255)

# Set the radius of the molecules
radius = 10

# Create a list to store the molecules
molecules = []

# Create 10 molecules with random positions and velocities
for i in range(10):
    x = random.randint(radius, window_size[0] - radius)
    y = random.randint(radius, window_size[1] - radius)
    vx = random.uniform(-1, 1)
    vy = random.uniform(-1, 1)
    molecules.append([x, y, vx, vy])

# Set the frame rate
clock = pygame.time.Clock()
frame_rate = 60

# Run the simulation
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the positions of the molecules
    for molecule in molecules:
        molecule[0] += molecule[2]
        molecule[1] += molecule[3]

        # Check if the molecule has reached the edge of the window and reverse its velocity if necessary
        if molecule[0] < radius or molecule[0] > window_size[0] - radius:
            molecule[2] = -molecule[2]
        if molecule[1] < radius or molecule[1] > window_size[1] - radius:
            molecule[3] = -molecule[3]

    # Clear the screen
    screen.fill(white)

    # Draw the molecules
    for molecule in molecules:
        pygame.draw.circle(screen, black, (int(molecule[0]), int(molecule[1])), radius)

    # Update the display
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(frame_rate)

# Quit Pygame
pygame.quit()
