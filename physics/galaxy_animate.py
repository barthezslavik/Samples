import pygame
import numpy as np

# Set the size of the grid and the number of particles
N = 100
num_particles = 100

# Initialize the pygame window
screen_size = (500, 500)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Galaxy with Inverted Entropic Force')

# Create a grid of points representing the positions of the particles
X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

# Calculate the entropic force at each point using a simple function
F = -np.exp(-X**2 - Y**2)

# Initialize the positions and velocities of the particles
positions = np.zeros((num_particles, 2))
velocities = np.zeros((num_particles, 2))

# Set the initial positions of the particles randomly within the grid
for i in range(num_particles):
    positions[i, 0] = np.random.uniform(-1, 1)
    positions[i, 1] = np.random.uniform(-1, 1)

# Set the rotation speed of the particles
rotation_speed = 0.1

# Set the size and color of the particles
particle_size = 5
particle_color = (0, 0, 0)

# Set the running flag and the clock
running = True
clock = pygame.time.Clock()

# Main game loop
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the positions of the particles based on their velocities
    positions += velocities

    # Create an array of indices for indexing into the F array
    indices = np.round(positions * (N-1)).astype(int)
    X_indices, Y_indices = np.meshgrid(indices[:, 0], indices[:, 1], indexing='ij')
    indices = np.stack((X_indices, Y_indices), axis=2)
    
    # Update the velocities of the particles based on the entropic force
    # velocities += [0.1, -0.1]#F[indices[:,:,0], indices[:,:,1]]
    
    # Rotate the positions of the particles around the center of the galaxy
    positions = np.dot(positions, [[np.cos(rotation_speed), -np.sin(rotation_speed)],
                                   [np.sin(rotation_speed), np.cos(rotation_speed)]])
    
    # Clear the screen
    screen.fill((255, 255, 255))
    
    # Draw the particles on the screen
    for i in range(num_particles):
        pygame.draw.circle(screen, particle_color, (int((positions[i, 0] + 1) * screen_size[0] / 2),
                                                    int((positions[i, 1] + 1) * screen_size[1] / 2)),
                                                    particle_size)
    
    # Update the display
    pygame.display.flip()
    
    # Limit the frame rate
    clock.tick(60)

# Quit pygame
pygame.quit()
