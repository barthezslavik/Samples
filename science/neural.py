import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the window
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)

# Define the colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)

# Define the neural network parameters
input_layer_size = 2
hidden_layer_size = 3
output_layer_size = 1

# Define the weights
Theta1 = np.random.randn(input_layer_size, hidden_layer_size)
Theta2 = np.random.randn(hidden_layer_size, output_layer_size)

# Define the sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the feedforward propagation function
def forward_propagation(X, Theta1, Theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, 1, axis=1)
    z2 = a1 @ Theta1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = a2 @ Theta2.T
    h = sigmoid(z3)
    return h.reshape(m,)

# Define the main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(white)

    # Draw the neural network
    pygame.draw.circle(screen, black, (200, 300), 20, 0)
    pygame.draw.circle(screen, black, (600, 300), 20, 0)
    pygame.draw.line(screen, black, (220, 300), (580, 300), 3)
    for i in range(hidden_layer_size):
        pygame.draw.circle(screen, red, (400, 200 + i * 100), 20, 0)
        pygame.draw.line(screen, black, (220, 300), (400, 200 + i * 100), 3)
        pygame.draw.line(screen, black, (400, 200 + i * 100), (580, 300), 3)

    # Use the neural network to make a prediction
    X = np.array([[1, 2]])
    y_pred = forward_propagation(X, Theta1, Theta2)
    print(y_pred)

    # Update the screen
    pygame.display.update()

# Quit Pygame
pygame.quit()
