import numpy as np
import pygame
import sys

pygame.init()

particles = 100
x, y = np.zeros((particles)), np.zeros((particles))

def update_screen(screen, x, y):
    screen.fill((255, 255, 255))
    for i in range(particles):
        pygame.draw.circle(screen, (0, 0, 0), (int(x[i]), int(y[i])), 1)
    pygame.display.update()

screen = pygame.display.set_mode((400, 400))

while True:
    for i in range(particles):
        x[i] = x[i] + np.random.normal(0, 1)
        y[i] = y[i] + np.random.normal(0, 1)
        x[i] = min(max(0, x[i]), 400)
        y[i] = min(max(0, y[i]), 400)
    update_screen(screen, x, y)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

pygame.quit()
sys.exit()
