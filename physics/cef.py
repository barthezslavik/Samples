import pygame
import random

# Initialize Pygame
pygame.init()

# Set the window size
window_size = (600, 600)

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the background color
bg_color = (255, 255, 255)

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
        pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), radius)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
