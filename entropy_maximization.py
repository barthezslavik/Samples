import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Set the window size
window_size = (600, 600)

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the background color
bg_color = (255, 255, 255)

# Set the number of molecules
num_molecules = 50

# Set the initial temperature of the system
temperature = 300  # Kelvin

# Set the mass of the molecules
mass = 1  # kilograms

# Set the radius of the molecules
radius = 10

# Set the initial positions and velocities of the molecules
molecules = []
for i in range(num_molecules):
    x = random.randint(radius, window_size[0] - radius)
    y = random.randint(radius, window_size[1] - radius)
    vx = random.gauss(0, math.sqrt(temperature / mass))
    vy = random.gauss(0, math.sqrt(temperature / mass))
    molecules.append((x, y, vx, vy))

# Set the time step
dt = 0.1  # seconds

# Set the coefficient of restitution (bouncing)
e = 0.9

# Set the simulation running flag
running = True

# Run the simulation loop
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the positions and velocities of the molecules
    for i, molecule in enumerate(molecules):
        x, y, vx, vy = molecule
        x += vx * dt
        y += vy * dt

        # Check for collisions with the walls of the container
        if x < radius or x > window_size[0] - radius:
            vx *= -e
        if y < radius or y > window_size[1] - radius:
            vy *= -e

        # Update the molecule's position and velocity
        molecules[i] = (x, y, vx, vy)

    # Clear the screen
    screen.fill(bg_color)

    # Draw the molecules
    for x, y, _, _ in molecules:
        pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), radius)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
