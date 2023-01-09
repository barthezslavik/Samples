import numpy as np
import matplotlib.pyplot as plt

# Set the size of the grid
N = 100

# Create a grid of points representing the positions of the particles
X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))

# Calculate the entropic force at each point using a simple function
F = -np.exp(-X**2 - Y**2)

# Plot the entropic force as a color map
plt.imshow(F, cmap='Blues')
plt.colorbar()
plt.show()
