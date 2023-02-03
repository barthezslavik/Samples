import numpy as np
import pygame
import sys

pygame.init()

def LJ(r):
    return 4 * (r**-12 - r**-6)

particles = 100
x, y = np.random.rand(particles) * 400, np.random.rand(particles) * 400
vx, vy = np.zeros((particles)), np.zeros((particles))
fx, fy = np.zeros((particles)), np.zeros((particles))

def update_forces(x, y, fx, fy):
    for i in range(particles):
        fx[i], fy[i] = 0, 0
        for j in range(i + 1, particles):
            r = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            f = LJ(r)
            fx[i] = fx[i] + f * (x[i] - x[j]) / r
            fy[i] = fy[i] + f * (y[i] - y[j]) / r
            fx[j] = fx[j] - f * (x[i] - x[j]) / r
            fy[j] = fy[j] - f * (y[i] - y[j]) / r

def update_screen(screen, x, y):
    screen.fill((255, 255, 255))
    for i in range(particles):
        pygame.draw.circle(screen, (0, 0, 0), (int(x[i]), int(y[i])), 2)
    pygame.display.update()

screen = pygame.display.set_mode((400, 400))

while True:
    update_forces(x, y, fx, fy)
    for i in range(particles):
        x[i] = x[i] + vx[i] + 0.5 * fx[i]
        y[i] = y[i] + vy[i] + 0.5 * fy[i]
        x[i] = min(max(0, x[i]), 400)
        y[i] = min(max(0, y[i]), 400)
        vx[i] = vx[i] + fx[i]
        vy[i] = vy[i] + fy[i]
    update_screen(screen, x, y)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

pygame.quit()
sys.exit()
