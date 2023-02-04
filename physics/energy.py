import pygame
import numpy as np

# Constants
MASS = 100  # particle mass
L = 400  # width of box
L_OVER_5 = L / 5  # height of box
TAU = 10  # causal entropic force parameter
T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force
TIMESTEP = 0.025  # time step for simulation

# Initialize Pygame
pygame.init()

# Create a Clock object to control the frame rate
clock = pygame.time.Clock()

# Set up screen
screen = pygame.display.set_mode((L, L_OVER_5))
pygame.display.set_caption("Particle in a Box")

# Initialize particle position and momentum
position = np.array([L / 10, L_OVER_5 / 10])
momentum = np.array([0, 0], dtype=np.float64)

# Set up font for displaying text
font = pygame.font.Font(None, 20)

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
    # causal_entropic_force = np.random.normal(0, np.sqrt(T_C / TIMESTEP), size=2)
    s = -(position**2 + momentum**2) / (2 * T_R)  # entropy of system
    grad_s = np.array([-position, -momentum])  # gradient of entropy
    causal_entropic_force = T_R * grad_s / TAU

    # Calculate total force
    total_force = energetic_force + random_force + causal_entropic_force[0]

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

    causal = font.render(str(causal_entropic_force), True, (0, 0, 0))
    # screen.blit(causal, (10, 10))

    random = font.render(str(random_force), True, (0, 0, 0))
    # screen.blit(random, (10, 30))

    # Draw particle
    pygame.draw.circle(screen, (0, 0, 0), position.astype(int), 2)

    pygame.display.flip()
    clock.tick(10)

# Quit Pygame
pygame.quit()