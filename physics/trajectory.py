import numpy as np
import matplotlib.pyplot as plt

T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force

# Set up grid
X, Y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))

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
    if n == 5000:
        break

    # Update position
    position += velocity

    # Find indices of closest known x- and y-values
    i_x = np.where(X[0,:] <= position[0])[0][-1]
    i_y = np.where(Y[:,0] <= position[1])[0][-1]

    # Set velocity based on known F_x and F_y values
    velocity = np.array([F_x[i_y, i_x], F_y[i_y, i_x]]) * 1000

    n += 1

    # Draw particle
    plt.plot(position[0], position[1], 'k.')

# Visualize gradient
plt.quiver(X, Y, F_x, F_y)
plt.show()
