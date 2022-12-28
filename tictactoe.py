# Import the necessary modules and libraries
import pygame

# Define the grid size and cell size
GRID_SIZE = 3
CELL_SIZE = 100

# Define the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define the players
PLAYER_X = "X"
PLAYER_O = "O"

# Define the grid
grid = [[None, None, None],
        [None, None, None],
        [None, None, None]]

# Define the current player
player = PLAYER_X

# Initialize the game window
pygame.init()
window = pygame.display.set_mode((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
pygame.display.set_caption("Tic Tac Toe")

# Main game loop
while True:
    # Process input events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Get the clicked cell
            column = event.pos[0] // CELL_SIZE
            row = event.pos[1] // CELL_SIZE
            # Check if the cell is empty
            if grid[row][column] is None:
                # Place the current player's symbol in the cell
                grid[row][column] = player
                # Switch players
                player = PLAYER_O if player == PLAYER_X else PLAYER_X
    
    # Clear the game window
    window.fill(BLACK)

    # Draw the grid lines
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            pygame.draw.rect(window, WHITE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    # Draw the symbols
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == PLAYER_X:
                # Draw an X
                pygame.draw.line(window, WHITE, (x * CELL_SIZE + 10, y * CELL_SIZE + 10), ((x + 1) * CELL_SIZE - 10, (y + 1) * CELL_SIZE - 10), 2)
                pygame.draw.line(window, WHITE, (x * CELL_SIZE + 10, (y + 1) * CELL_SIZE - 10), ((x + 1) * CELL_SIZE - 10, y * CELL_SIZE + 10), 2)
            elif grid[y][x] == PLAYER_O:
                # Draw an O
                pygame.draw.circle(window, WHITE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2 - 10, 2)

    # Update and render the game window
    pygame.display.update()
