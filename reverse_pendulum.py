import pygame
import math
import random

# Constants
GRAVITY = 9.81  # m/s^2
MASS_PENDULUM = 1.0  # kg
LENGTH_PENDULUM = 1.0  # m
DAMPING = 0.1  # N*s/m
DT = 0.01  # s

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 640, 480
screen = pygame.display.set_mode((width, height))

# Set up the pendulum
x = width / 2
y = height / 2
angle = math.pi / 2
angular_velocity = 0.0

# Set up the entropic force
entropic_force = 0.0

# Set up the clock
clock = pygame.time.Clock()

# Main loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Update the entropic force
    entropic_force = random.uniform(-1.0, 1.0)

    # Update the angular velocity and angle
    angular_acceleration = -GRAVITY / LENGTH_PENDULUM * math.sin(angle) - DAMPING * angular_velocity + entropic_force
    angular_velocity += angular_acceleration * DT
    angle += angular_velocity * DT

    # Update the position of the pendulum
    x = width / 2 + LENGTH_PENDULUM * math.sin(angle)
    y = height / 2 + LENGTH_PENDULUM * math.cos(angle)

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the pendulum
    pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 10)
    pygame.draw.line(screen, (0, 0, 0), (width / 2, height / 2), (int(x), int(y)))

    # Update the display
    pygame.display.flip()

    # Delay to maintain a constant frame rate
    clock.tick(60)
