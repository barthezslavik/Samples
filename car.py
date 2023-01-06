import pygame
import numpy as np

# Constants
WIDTH = 800  # width of screen
HEIGHT = 600  # height of screen
MASS = 10  # mass of car
TAU = 10  # causal entropic force parameter
T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force
TIMESTEP = 0.1  # time step for simulation

# Initialize Pygame
pygame.init()

# Set up screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Moving on a Flat Surface")

# Initialize car position and velocity
position = np.array([WIDTH / 2, HEIGHT / 2])
velocity = np.array([0, 0], dtype=np.float64)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate energetic force components
    energetic_force = -velocity / TAU

    # Calculate random force components
    random_force = np.random.normal(0, np.sqrt(MASS * T_R / TIMESTEP), size=2)

    # Calculate causal entropic force components
    s = -(position**2 + velocity**2).sum() / (2 * T_R)  # entropy of system
    grad_s = -position / T_R  # gradient of entropy
    causal_entropic_force = T_R * grad_s / TAU

    # Calculate total force
    total_force = energetic_force + random_force + causal_entropic_force

    # Update velocity
    velocity += total_force * TIMESTEP

    # Update position
    position += velocity * TIMESTEP

    # Check for collisions with walls
    if position[0] < 0:
        position[0] = 0
        velocity[0] *= -1
    if position[0] > WIDTH:
        position[0] = WIDTH
        velocity[0] *= -1
    if position[1] < 0:
        position[1] = 0
        velocity[1] *= -1
    if position[1] > HEIGHT:
        position[1] = HEIGHT
        velocity[1] *= -1

    # Fill screen with white
    screen.fill((255, 255, 255))

    # Draw car
    car_width = 20
    car_height = 30
    pygame.draw.rect(screen, (0, 0, 0), (position[0] - car_width / 2, position[1] - car_height / 2, car_width, car_height))

    pygame.display.flip()
