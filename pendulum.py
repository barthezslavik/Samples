import pygame
import numpy as np

# Set up the Pygame window
WIDTH = 800
HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force

# Set up grid
X, Y = np.meshgrid(np.linspace(-WIDTH/2, WIDTH/2, 30), np.linspace(-HEIGHT/2, HEIGHT/2, 30))

# Calculate gradient of causal entropic force
F_x = -X / T_R * np.sqrt(X**2 + Y**2)  # x-component of gradient
F_y = -Y / T_R * np.sqrt(X**2 + Y**2)  # y-component of gradient

# Set up pendulum
m = 1  # mass of pendulum bob
L = 100  # length of pendulum
g = 9.81  # gravitational acceleration
dt = 0.1  # time step

# Initialize pendulum position and velocity
theta = np.pi / 4  # initial angle of pendulum
omega = 0  # initial angular velocity

# Main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate potential energy landscape
    U = m * g * L * (1 - np.cos(theta))
    # Calculate gradient of potential energy landscape
    F = -m * g * L * np.sin(theta)
    # Calculate acceleration
    a = -(g/L) * np.sin(theta)
    # Update angular velocity
    omega += a * dt
    # Update angle
    theta += omega * dt

    # Calculate interpolated values of F_x and F_y at current position
    F_x_interp = np.interp(theta, X[0,:], F_x[:,0])
    F_y_interp = np.interp(theta, Y[:,0], F_y[0,:])
    # Set velocity based on interpolated values
    velocity = np.array([F_x_interp, F_y_interp]) * 1000

    # Update position
    position = np.array([L * np.sin(theta), L * np.cos(theta)])

    # Draw the pendulum
    screen.fill((255, 255, 255))
    pygame.draw.line(screen, (0, 0, 0), (WIDTH/2, HEIGHT/2), (WIDTH/2 + position[0], HEIGHT/2 + position[1]), 1)
    pygame.draw.circle(screen, (0, 0, 0), (int(WIDTH/2 + position[0]), int(HEIGHT/2 + position[1])), 5)

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Shut down Pygame
pygame.quit()