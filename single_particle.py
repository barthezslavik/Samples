import pygame
import numpy as np

# Set up the Pygame window
WIDTH = 800
HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force

# Set up grid
X, Y = np.meshgrid(np.linspace(-WIDTH/2, WIDTH/2, 30), np.linspace(-HEIGHT/2, HEIGHT/2, 30))

# Calculate gradient of causal entropic force
F_x = -X / T_R  # x-component of gradient
F_y = -Y / T_R  # y-component of gradient

# Initialize particle position and velocity
position = np.array([np.random.choice(X.flatten()), np.random.choice(Y.flatten())])

# Calculate interpolated values of F_x and F_y at current position
F_x_interp = np.interp(position[0], X[0,:], F_x[:,0])
F_y_interp = np.interp(position[1], Y[:,0], F_y[0,:])

# Set velocity based on interpolated values
velocity = np.array([F_x_interp, F_y_interp])
n = 0

# Run the simulation
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update position
    position += velocity

    # Find indices of closest known x- and y-values
    i_x = np.where(X[0,:] <= position[0])[0][-1]
    i_y = np.where(Y[:,0] <= position[1])[0][-1]

    # Set velocity based on known F_x and F_y values
    velocity = np.array([F_x[i_y, i_x], F_y[i_y, i_x]]) * 1000

    # Draw the particle
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (0, 0, 0), (int(position[0] + WIDTH/2), int(position[1] + HEIGHT/2)), 5)

    # Visualize gradient
    # for x, y, f_x, f_y in zip(X.flatten(), Y.flatten(), F_x.flatten(), F_y.flatten()):
    #    pygame.draw.line(screen, (0, 0, 0), (x + WIDTH/2, y + HEIGHT/2), (x + f_x + WIDTH/2, y + f_y + HEIGHT/2))

    # Update the display
    pygame.display.flip()

# Shut down Pygame
pygame.quit()