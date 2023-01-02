import pygame
import random
import math

# Set up the Pygame window
WIDTH = 800
HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the particle
x = random.uniform(0, WIDTH)
y = random.uniform(0, HEIGHT)
vx = random.uniform(-100, 100)
vy = random.uniform(-100, 100)

# Set up the simulation parameters
dt = 0.1  # Time step
T = 10  # Time horizon
K = 0.1  # Entropic force coefficient

# Set up the boundaries of the box
left = 0
right = WIDTH
top = 0
bottom = HEIGHT

# Run the simulation
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate the expected future position of the particle
    x_future = x + vx * T
    y_future = y + vy * T

    # Calculate the entropic force towards the expected future position
    dx = x_future - x
    dy = y_future - y
    r = math.sqrt(dx**2 + dy**2)
    Fx = K * dx / r
    Fy = K * dy / r

    # Update the velocity of the particle
    vx += Fx * dt
    vy += Fy * dt
    # Update the position of the particle
    x += vx * dt
    y += vy * dt

    # Check for collisions with the boundaries of the box
    if x < left:
        x = left
        vx = -vx
    elif x > right:
        x = right
        vx = -vx
    if y < top:
        y = top
        vy = -vy
    elif y > bottom:
        y = bottom
        vy = -vy

    # Draw the particle
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 20)

    # Update the display
    pygame.display.flip()

# Shut down Pygame
pygame.quit()