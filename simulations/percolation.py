import numpy as np
import pygame
import sys

pygame.init()

def percolation(p, N):
    grid = np.random.choice([0, 1], size=(N, N), p=[1-p, p])
    return grid

def update_screen(screen, grid):
    screen.fill((255, 255, 255))
    for i in range(N):
        for j in range(N):
            if grid[i, j] == 1:
                pygame.draw.rect(screen, (0, 0, 0), (i * 10, j * 10, 10, 10))
    pygame.display.update()

N = 100
p = 0.5
grid = percolation(p, N)

screen = pygame.display.set_mode((N * 10, N * 10))

while True:
    update_screen(screen, grid)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

pygame.quit()
sys.exit()
