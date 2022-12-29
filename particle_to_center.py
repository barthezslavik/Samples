import pygame
import random
import numpy as np
from scipy.constants import Boltzmann
from scipy.stats import norm
from math import sqrt

# Initialize Pygame
pygame.init()

# Set the window size
window_size = (400, 200)

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the background color
bg_color = (255, 255, 255)

class particleBox:

    def __init__(self):
        # state variables
        self.length = 400.0
        self.start = np.array([self.length/10.0, self.length/10.0])
        self.bounds = ((0.0, self.length), (0.0, self.length/5.0))
        self.DIMS = len(self.bounds)

        self.KB = Boltzmann      # Boltzmann Constant
        self.TAU = 10.0          # Time horizon
        self.TR = 400000.0       # Temperature of random movement
        self.TC = 5.0 * self.TR  # Causal Path Temperature
        self.TIMESTEP = 0.05     # Interval between random walk sampling
        self.MASS = 10.0 ** -21

        self.MEAN = 0.0
        self.AMPLITUDE = sqrt(self.MASS * self.KB * self.TR) / self.TIMESTEP
        self.DISTRIBUTION = norm(0.0, 1.0)

    def step_microstate(self, cur_state):
        'compute next distance by Forward Euler'
        random = self.DISTRIBUTION.rvs(self.DIMS)
        force = self.AMPLITUDE * random + self.MEAN
        euler = (self.TIMESTEP ** 2.0) / (2.0 * self.MASS)
        constant = 2.0 / self.TIMESTEP
        pos = cur_state + force * euler * constant
        return pos, force

    def valid(self, walk, position):
        'determine whether a walk is valid'
        if ((position[0] < self.bounds[0][0]) or
            (position[0] > self.bounds[0][1]) or
            (position[1] < self.bounds[1][0]) or
            (position[1] > self.bounds[1][1])):
            return False
        else:
            return True

    def step_macrostate(self, cur_macrostate, causal_entropic_force):
        'move the particle subject to causal_entropic_force'
        euler = (self.TIMESTEP ** 2.0) / (4.0 * self.MASS)
        distance = causal_entropic_force * euler
        return cur_macrostate + distance

    def draw(self, current):
        'draw the current position of the particle on the Pygame window'
        x, y = current
        pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 5)
        pygame.display.flip()

# Create the particle box
pb = particleBox()

# Set the number of particles
num_particles = 50

# Set the initial positions and velocities of the particles
particles = []
for i in range(num_particles):
    x = random.randint(0, window_size[0])
    y = random.randint(0, window_size[1])
    vx = random.uniform(-1, 1)
    vy = random.uniform(-1, 1)
    particles.append((x, y, vx, vy))

# Set the time step
dt = 0.1  # seconds

# Set the particle radius
radius = 5

# Set the simulation running flag
running = True

# Run the simulation loop
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the positions of the particles
    for i in range(num_particles):
        x, y, vx, vy = particles[i]
        x += vx * dt
        y += vy * dt
        particles[i] = (x, y, vx, vy)

    # Draw the particles on the screen
    screen.fill(bg_color)
    for x, y, _, _ in particles:
        pb.draw((x, y))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()