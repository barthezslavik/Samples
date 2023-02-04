import pygame

# Constants
WIDTH = 400
HEIGHT = 300
MASS = 1  # kg
K = 1  # N/m

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Set up the particle
particle_pos = [100, 100]  # x, y position (pixels)
# particle_vel = [10, 5]  # x, y velocity (pixels/s)
# particle_mass = MASS  # kg

# Define the particle's momentum
particle_momentum = [5, 5]  # x and y components of momentum

# Define the particle's mass
particle_mass = MASS  # in kilograms

# Define the particle's energy
particle_energy = 0  # initialize to zero

# Set up the box
box_size = [WIDTH, HEIGHT]  # width, height (pixels)
box_color = (255, 255, 255)  # gray

# Main loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:   
            pygame.quit()
            sys.exit()

    # Update the particle's position using its momentum and energy
    particle_pos[0] += particle_momentum[0] / particle_mass
    particle_pos[1] += particle_momentum[1] / particle_mass

    # Calculate the potential energy of the particle
    particle_pe = K * (particle_pos[0]**2 + particle_pos[1]**2) / 2

    # Calculate the kinetic energy of the particle
    particle_ke = 0.5 * particle_mass * ((particle_momentum[0] / particle_mass)**2 + (particle_momentum[1] / particle_mass)**2)

    # Calculate the total energy of the particle
    particle_energy = particle_pe + particle_ke

    # Check if the particle has collided with the left or right walls
    if particle_pos[0] < 0 or particle_pos[0] > box_size[0]:
        # Reverse the x component of the particle's momentum
        particle_momentum[0] = -particle_momentum[0]

    # Check if the particle has collided with the top or bottom walls
    if particle_pos[1] < 0 or particle_pos[1] > box_size[1]:
        # Reverse the y component of the particle's momentum
        particle_momentum[1] = -particle_momentum[1]

    # Draw the box and the particle
    screen.fill((255, 255, 255))  # white background
    pygame.draw.rect(screen, box_color, (0, 0, box_size[0], box_size[1]))  # draw box
    pygame.draw.circle(screen, (0, 0, 0), particle_pos, 5)  # draw particle

    # Update the display
    pygame.display.flip()
    clock.tick(60)  # limit to 60 fps
