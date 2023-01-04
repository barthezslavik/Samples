import pygame
import numpy as np

# Constants
MASS = 1e-21  # particle mass
L = 400  # width of box
L_OVER_5 = L / 5  # height of box
TAU = 10  # causal entropic force parameter
T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force
TIMESTEP = 0.025  # time step for simulation

# Initialize Pygame
pygame.init()

# Set up screen
screen = pygame.display.set_mode((L, L_OVER_5))
pygame.display.set_caption("Particle in a Box")

# Initialize particle position and momentum
position = np.array([L / 10, L_OVER_5 / 10])
momentum = np.array([0, 0])

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate energetic force components
    energetic_force = -momentum / TAU

    # Calculate random force components
    random_force = np.random.normal(0, np.sqrt(MASS * T_R / TIMESTEP), size=2)

    # Calculate causal entropic force components
    causal_entropic_force = np.random.normal(0, np.sqrt(T_C / TIMESTEP), size=2)

    # Calculate total force
    total_force = energetic_force + random_force + causal_entropic_force
    # print(total_force * TIMESTEP)

    # Update momentum
    momentum += total_force * TIMESTEP

    # Update position
    position += momentum / MASS * TIMESTEP

    # Check for collisions with walls
    if position[0] < 0 or position[0] > L:
        momentum[0] *= -1
    if position[1] < 0 or position[1] > L_OVER_5:
        momentum[1] *= -1

    # Fill screen with white
    screen.fill((255, 255, 255))

    # Draw particle as white fill with black border
    pygame.draw.circle(screen, (0, 0, 0), position.astype(int), 5, 5)

    pygame.display.flip()

# Quit Pygame
pygame.quit()