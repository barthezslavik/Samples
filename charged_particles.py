import pygame
import numpy as np

# Set up the Pygame window
pygame.init()
screen = pygame.display.set_mode((600, 600))

# Define the constants for the simulation
NUM_PARTICLES = 20
CHARGE_MULTIPLIER = 1.0
GRAVITY = 0.1
DT = 1

# Generate the initial positions and charges for the particles
positions = np.random.uniform(0, 600, (NUM_PARTICLES, 2))
velocities = np.random.uniform(low=-1, high=1, size=(NUM_PARTICLES, 2))
charges = np.random.uniform(-1, 1, NUM_PARTICLES)

# Set up the Pygame clock to control the frame rate
clock = pygame.time.Clock()

# Run the simulation loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    # Update the positions and velocities of the particles
    for i in range(NUM_PARTICLES):
        # Calculate the net force on the particle
        force = np.array([0.0, 0.0])
        for j in range(NUM_PARTICLES):
            if i == j:
                continue
            diff = positions[j] - positions[i]
            distance = np.linalg.norm(diff)
            force += CHARGE_MULTIPLIER * charges[i] * charges[j] * diff / (distance**3)
        force += np.array([0.0, GRAVITY])
        
        # Update the velocity and position of the particle using Euler's method
        positions[i] = positions[i] + DT * velocities[i]
        
        # Handle collisions with the walls of the box
        if positions[i, 0] < 0 or positions[i, 0] > 600:
            velocities[i, 0] *= -1
        if positions[i, 1] < 0 or positions[i, 1] > 600:
            velocities[i, 1] *= -1
    
    # Clear the screen
    screen.fill((0, 0, 0))
    
    # Draw the particles
    for i in range(NUM_PARTICLES):
        if charges[i] > 0:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        pygame.draw.circle(screen, color, (int(positions[i, 0]), int(positions[i, 1])), 10)
    
    # Update the display
    pygame.display.flip()
    
    # Limit the frame rate to 60 FPS
    clock.tick(60)
