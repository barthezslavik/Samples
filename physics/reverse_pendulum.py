import pygame
import math

# Set up the Pygame window
WIDTH = 800
HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the pendulum
L = 200  # Length of the pendulum
theta = math.pi / 4  # Angle of the pendulum
omega = 0  # Angular velocity of the pendulum

# Set up the simulation parameters
dt = 1  # Time step
g = 9.81  # Acceleration due to gravity
K = 0.1  # Entropic force coefficient
T = 10  # Time horizon

# Set up the pivot point
pivot_x = WIDTH / 2
pivot_y = HEIGHT / 2

# Run the simulation
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate the expected future position of the pendulum
    x_future = pivot_x + L * math.sin(theta + omega * T)
    y_future = pivot_y + L * math.cos(theta + omega * T)

    # Calculate the entropic force towards the expected future position
    dx = x_future - pivot_x
    dy = y_future - pivot_y
    r = math.sqrt(dx**2 + dy**2)
    Fx = K * dx / r
    Fy = K * dy / r

    # Calculate the acceleration of the pendulum
    alpha = (Fx * math.cos(theta) + Fy * math.sin(theta)) / L
    # Update the angular velocity of the pendulum
    omega += alpha * dt
    # Update the angle of the pendulum
    theta += omega * dt

    # Draw the pendulum
    screen.fill((255, 255, 255))
    x = pivot_x + L * math.sin(theta)
    y = pivot_y + L * math.cos(theta)
    pygame.draw.line(screen, (0, 0, 0), (pivot_x, pivot_y), (x, y), 5)
    pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 10)

    # Update the display
    pygame.display.flip()

# Shut down Pygame
pygame.quit()
