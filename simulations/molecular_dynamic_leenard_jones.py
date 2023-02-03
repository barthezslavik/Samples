import numpy as np
import pygame
import sys

pygame.init()

steps = 1000000
particles = 100
x, y = np.zeros((steps, particles)), np.zeros((steps, particles))

def update_screen(screen, x, y, t):
    screen.fill((255, 255, 255))
    for i in range(particles):
        pygame.draw.circle(screen, (0, 0, 0), (int(x[t, i]), int(y[t, i])), 1)
    pygame.display.update()

screen = pygame.display.set_mode((400, 400))

for t in range(1, steps):
    x[t, :] = x[t - 1, :] + np.random.normal(0, 1, particles)
    y[t, :] = y[t - 1, :] + np.random.normal(0, 1, particles)
    update_screen(screen, x, y, t)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

pygame.quit()
sys.exit()
