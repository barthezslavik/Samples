import pygame
import random
import time

def change_direction(direction):
    """Change the direction of the particle.

    Parameters:
        direction (str): The current direction of the particle.

    Returns:
        str: The new direction of the particle.
    """
    # Choose a new direction based on the current direction
    if direction == "up":
        return "left"
    elif direction == "left":
        return "down"
    elif direction == "down":
        return "right"
    elif direction == "right":
        return "up"


def calculate_entropic_force(maze, x, y):
    """Calculate the entropic force at the specified position in the maze.

    Parameters:
        maze (list): The maze layout.
        x (int): The x-coordinate of the position.
        y (int): The y-coordinate of the position.

    Returns:
        float: The entropic force at the specified position.
    """
    # Set the initial entropic force to the minimum value
    force = min_force

    # Calculate the entropic force based on the surrounding cells in the maze
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # Skip the current position
            if dx == 0 and dy == 0:
                continue

            # Calculate the position of the surrounding cell
            cx = x + dx
            cy = y + dy

            # Skip the cell if it is out of bounds
            if cx < 0 or cy < 0 or cx >= len(maze[0]) or cy >= len(maze):
                continue

            # Increase the entropic force based on the value of the surrounding cell
            if maze[cy][cx] == 0:
                force += 0.1
            elif maze[cy][cx] == 1:
                force -= 0.1

    # Clamp the entropic force to the minimum and maximum values
    force = max(min_force, min(force, max_force))

    return force


# Define the maze layout
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Set the starting position of the particle
start_x = 1
start_y = 1

# Set the ending position of the particle
end_x = 8
end_y = 8

# Set the size of the cells in the maze (in pixels)
cell_size = 32

# Initialize Pygame
pygame.init()

# Set the window size and title
window_size = (len(maze[0]) * cell_size, len(maze) * cell_size)
pygame.display.set_caption("Entropic Force Maze")

# Create the window
screen = pygame.display.set_mode(window_size)

# Set the background color
bg_color = (255, 255, 255)

# Set the colors for the walls, open spaces, and goals
wall_color = (0, 0, 0)
open_color = (255, 255, 255)
goal_color = (0, 255, 0)

# Set the color for the particle
particle_color = (255, 0, 0)

# Set the initial position of the particle
particle_x = start_x
particle_y = start_y

# Set the size of the particle (in pixels)
particle_size = 16

# Set the movement speed of the particle (in pixels per frame)
movement_speed = 4

# Set the initial direction of the particle
direction = "right"

# Set the initial entropic force of the particle
entropic_force = 0

# Set the threshold for the entropic force (below which the particle will change direction)
force_threshold = 0.5

# Set the minimum and maximum values for the entropic force
min_force = 0
max_force = 1

# Set the flag to indicate whether the particle has reached the goal
reached_goal = False

# Start the game loop
while True:
    # Check for quit events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Update the entropic force based on the current position of the particle
    entropic_force = calculate_entropic_force(maze, particle_x, particle_y)

    # If the entropic force is below the threshold, change the direction of the particle
    if entropic_force < force_threshold:
        direction = change_direction(direction)

    # Update the position of the particle based on the current direction
    particle_x, particle_y = move_particle(particle_x, particle_y, direction, movement_speed)

    # Check if the particle has reached the goal
    if particle_x == end_x and particle_y == end_y:
        reached_goal = True
        break

    # Draw the maze and the particle
    draw_maze(screen, maze, cell_size, wall_color, open_color, goal_color)
    draw_particle(screen, particle_x, particle_y, particle_size, particle_color)

    # Update the display
    pygame.display.flip()

    # Wait for a short time
    time.sleep(0.01)

# If the particle has reached the goal, display a message
if reached_goal:
    print("The particle reached the goal!")
else:
    print("The particle did not reach the goal.")
