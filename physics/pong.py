import pygame
import numpy as np

# Constants
L = 400  # width of game screen
H = 300  # height of game screen
PADDLE_WIDTH = 20  # width of paddle
PADDLE_HEIGHT = 80  # height of paddle
TIMESTEP = 0.025  # time step for simulation
TAU = 10  # causal entropic force parameter
T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force

# Initialize Pygame
pygame.init()

# Create a Clock object to control the frame rate
clock = pygame.time.Clock()

# Set up screen
screen = pygame.display.set_mode((L, H))
pygame.display.set_caption("Pong")

# Initialize paddle position and velocity
paddle_pos = np.array([L / 2, H / 2])
paddle_vel = np.array([0, 0], dtype=np.float64)

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate energetic force
    energetic_force = -paddle_vel / TAU

    # Calculate random force
    random_force = np.random.normal(0, np.sqrt(T_R / TIMESTEP), size=2)

    # Calculate entropy of system
    k = 1  # Boltzmann constant
    p = paddle_pos / (L * H)  # probability distribution of paddle position
    # s = -k * np.log(p)  # entropy of system

    # Calculate gradient of entropy
    grad_s = -k * (1 / p)  # gradient of entropy

    # Calculate causal entropic force
    causal_entropic_force = T_R * grad_s / TAU

    # Calculate total force
    total_force = energetic_force + random_force + causal_entropic_force

    # Update paddle velocity
    paddle_vel += total_force * TIMESTEP

    # Update paddle position
    paddle_pos += paddle_vel * TIMESTEP

    # Check for collisions with walls
    if paddle_pos[0] < 0 or paddle_pos[0] > L - PADDLE_WIDTH:
        paddle_vel[0] *= -1
    if paddle_pos[1] < 0 or paddle_pos[1] > H - PADDLE_HEIGHT:
        paddle_vel[1] *= -1

    # Fill screen with white
    screen.fill((255, 255, 255))

    # Draw paddle
    pygame.draw.rect(screen, (0, 0, 0), (paddle_pos[0], paddle_pos[1], PADDLE_WIDTH, PADDLE_HEIGHT))

    pygame.display.flip()
    clock.tick(1)

# Quit Pygame
pygame.quit()
