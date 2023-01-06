import numpy as np
import matplotlib.pyplot as plt

T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force

# Set up grid
X, Y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

# Calculate gradient of causal entropic force
F_x = -X / T_R  # x-component of gradient
F_y = -Y / T_R  # y-component of gradient

# Initialize particle position and velocity
position = np.array([np.random.choice(X.flatten()), np.random.choice(Y.flatten())])

# Calculate interpolated values of F_x and F_y at current position
F_x_interp = np.interp(position[0], X[0,:], F_x[:,0])
F_y_interp = np.interp(position[1], Y[:,0], F_y[0,:])

# Set velocity based on interpolated values
velocity = np.array([F_x_interp, F_y_interp])
n = 0

# Main loop
while True:
    if n == 50000:
        break
    # Update position
    position += velocity

    # Check if particle has left the grid
    # if (position[0] < 0) or (position[0] >= X.shape[0]) or (position[1] < 0) or (position[1] >= Y.shape[1]):
    #    break

    # Calculate interpolated values of F_x and F_y at current position
    F_x_interp = np.interp(position[0], X[0,:], F_x[:,0])
    F_y_interp = np.interp(position[1], Y[:,0], F_y[0,:])

    # Set velocity based on interpolated values
    velocity = np.array([F_x_interp, F_y_interp])

    n += 1

    # Draw particle
    # print(position)
    plt.plot(position[0], position[1], 'k.')

# Visualize gradient
plt.quiver(X, Y, F_x, F_y)
plt.show()
