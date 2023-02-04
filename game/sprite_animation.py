# Import the Pygame library and initialize the game engine
import random
import pygame
pygame.init()

# Create a Clock object to control the frame rate
clock = pygame.time.Clock()

# Define the dimensions of the game screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Create a Surface object to represent the game screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Create a Surface object to represent the animation
animation_surface = pygame.Surface((100, 100))

# Load the frames of the animation from image files
num_frames = 5
frames = []
for i in range(num_frames):
    frame = pygame.image.load(f"data/animations/frame_{i}.png")
    frames.append(frame)

# Define a variable to track the current frame of the animation
current_frame = 0
done = False

# Create a game loop to run until the user quits
while not done:
    # Check for events and handle them
    for event in pygame.event.get():
        # If the user closes the window, quit the game
        if event.type == pygame.QUIT:
            done = True

    # Update the current frame of the animation
    current_frame += 1
    if current_frame >= num_frames:
        current_frame = 0

    # Draw the current frame of the animation to the animation Surface
    animation_surface.blit(frames[current_frame], (0, 0))

    # Draw the animation Surface to the screen at a random position
    x = random.randint(0, SCREEN_WIDTH - 100)
    y = random.randint(0, SCREEN_HEIGHT - 100)
    screen.blit(animation_surface, (x, y))

    # Update the game screen and wait for the next frame
    pygame.display.flip()
    clock.tick(30)
