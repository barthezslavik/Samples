import numpy as np
import pygame
import sys

pygame.init()

L = 20
T = 5
np.random.seed(1234)
spin_grid = np.random.choice([-1, 1], size=(L, L))

def deltaE(S, i, j):
    return 2 * S[i, j] * (S[i, (j + 1) % L] + S[i, (j - 1 + L) % L] + S[(i + 1) % L, j] + S[(i - 1 + L) % L, j])

def update_screen(screen, spin_grid):
    screen.fill((255, 255, 255))
    for i in range(L):
        for j in range(L):
            if spin_grid[i, j] == 1:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)
            pygame.draw.rect(screen, color, (i * 20, j * 20, 20, 20))
    pygame.display.update()

screen = pygame.display.set_mode((400, 400))

for t in range(100000):
    i, j = np.random.randint(0, L, size=2)
    dE = deltaE(spin_grid, i, j)
    if dE <= 0 or np.random.rand() < np.exp(-dE / T):
        spin_grid[i, j] = -spin_grid[i, j]
    update_screen(screen, spin_grid)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

pygame.quit()
sys.exit()
