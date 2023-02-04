import pygame
import random

# Set up the Pygame window
WIDTH = 800
HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the particles
NUM_PARTICLES = 1000
particles = []
for i in range(NUM_PARTICLES):
    x = random.uniform(0, WIDTH)
    y = random.uniform(0, HEIGHT)
    vx = random.uniform(-100, 100)
    vy = random.uniform(-100, 100)
    particle = [x, y, vx, vy]
    particles.append(particle)

# Set up the simulation parameters
dt = 0.1  # Time step
K = 0.1  # Entropic force coefficient

# Run the simulation
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the particles
    for particle in particles:
        x, y, vx, vy = particle
        ax = -K * vx
        ay = -K * vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        particle[0] = x
        particle[1] = y
        particle[2] = vx
        particle[3] = vy

    # Calculate the instantaneous entropy production of the system
    # S = 0
    # for particle in particles:
    #     x, y, vx, vy = particle
    #     S += vx**2 + vy**2
    # S /= NUM_PARTICLES

    # Bias the particles towards maximum instantaneous entropy production
    for particle in particles:
        x, y, vx, vy = particle
        S = 1
        vx += random.gauss(0, S)
        vy += random.gauss(0, S)
        particle[2] = vx
        particle[3] = vy

    # Draw the particles
    screen.fill((255, 255, 255))
    for particle in particles:
        x, y, vx, vy = particle
        pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 2)

    # Update the display
    pygame.display.flip()

# Shut down Pygame
pygame.quit()
