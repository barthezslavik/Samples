import pygame
import numpy as np
from scipy.special import sph_harm

# Set the size of the window
WIDTH = 800
HEIGHT = 600

# Set the number of grid points in each dimension
NX = 100
NY = 100

# Set the maximum radius of the probability distribution
MAX_RADIUS = 5.0

# Set the number of quantum states to visualize
NUM_STATES = 5

# Set the color map
CMAP = pygame.color.THECOLORS

# Set the maximum probability
MAX_PROB = 4.0

# Initialize Pygame
pygame.init()

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set the title of the window
pygame.display.set_caption('Electron Cloud')

# Set the background color
screen.fill((255, 255, 255))

# Create the grid of points
X, Y = np.meshgrid(np.linspace(-MAX_RADIUS, MAX_RADIUS, NX), np.linspace(-MAX_RADIUS, MAX_RADIUS, NY))

# Compute the probability for each state
for n in range(NUM_STATES):
    for l in range(n):
        for m in range(-l, l+1):
            # Compute the probability at each point
            # prob = np.abs(sph_harm(m, l, 0, np.pi/2, np.arctan2(Y, X)))**2
            print(sph_harm(m, l, 0, np.pi/2, np.arctan2(Y, X)))
            #prob = (1,2,3)

            # Normalize the probability
            prob /= prob.max()

            # Set the color of each point based on the probability
            color = CMAP[f'{int(255*prob/MAX_PROB)}']

            # Draw the points
            for i in range(NX):
                for j in range(NY):
                    pygame.draw.circle(screen, color, (int((X[i,j]+MAX_RADIUS)/(2*MAX_RADIUS)*WIDTH), int((Y[i,j]+MAX_RADIUS)/(2*MAX_RADIUS)*HEIGHT)), int(prob[i,j]*10))

# Update the screen
pygame.display.flip()

# Run the Pygame loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()
