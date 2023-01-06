import pygame
import numpy as np

# Constants
MASS = 100  # mass of system
T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force
TAU = 10  # causal entropic force parameter
TIMESTEP = 0.025  # time step for simulation
L = 400  # width of box
L_OVER_5 = L / 5  # height of box

# Initialize Pygame
pygame.init()

# Set up screen
screen = pygame.display.set_mode((L, L_OVER_5))
pygame.display.set_caption("Inverted Pendulum and Cart Pole")

# Initialize position and momentum
position = np.array([L / 10, L_OVER_5 / 10])  # position of cart
momentum = np.array([0, 0], dtype=np.float64)  # momentum of cart
angle = np.pi / 4  # angle of pendulum
angular_momentum = np.float64(0)  # angular momentum of pendulum

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate causal entropic forces
    s = -(position**2 + momentum**2 + angle**2 + angular_momentum**2) / (2 * T_R)  # entropy of system
    grad_s = np.array([-position, -momentum, -angle, -angular_momentum])  # gradient of entropy
    causal_entropic_force = T_R * grad_s / TAU
    causal_entropic_torque = causal_entropic_force[2:]

    # Calculate random forces
    random_force = np.random.normal(0, np.sqrt(2 * MASS * T_R * TIMESTEP), size=2)
    random_torque = np.random.normal(0, np.sqrt(2 * MASS * T_R * TIMESTEP), size=2)

    # Calculate total forces and torques
    total_force = causal_entropic_force[:2] + random_force
    total_torque = causal_entropic_torque + random_torque

    # Update position and momentum
    position = (position + (momentum / MASS) * TIMESTEP + (total_force[0] / MASS) * TIMESTEP**2 / 2).astype('float64')
    momentum = (momentum + total_force[0] * TIMESTEP).astype('float64')
    angle += (angular_momentum / MASS) * TIMESTEP + (total_torque[0] / MASS) * TIMESTEP**2 / 2
    angular_momentum += total_torque[0] * TIMESTEP

    # Check for collisions with walls
    if position[0] < 0 or position[0] > L:
        momentum[0] *= -1
    if position[1] < 0 or position[1] > L_OVER_5:
        momentum[1] *= -1

    # Fill screen with white
    screen.fill((255, 255, 255))

    # Draw cart
    cart_x = int(position[0])
    cart_y = int(L_OVER_5 / 2)
    pygame.draw.rect(screen, (0, 0, 0), (cart_x - 10, cart_y - 5, 20, 10))

    # Draw pendulum
    pendulum_x = 1#int(cart_x + np.sin(angle[0]) * L_OVER_5 / 4)
    pendulum_y = 2#int(cart_y - np.cos(angle[1]) * L_OVER_5 / 4)
    pygame.draw.line(screen, (0, 0, 0), (cart_x, cart_y), (pendulum_x, pendulum_y), 2)
    pygame.draw.circle(screen, (0, 0, 0), (pendulum_x, pendulum_y), 5)

    pygame.display.flip()

# Quit Pygame
pygame.quit()
