import numpy as np
import matplotlib.pyplot as plt

T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force

# Set up grid
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 30))

# Calculate gradient of causal entropic force
F_x = -X / T_R  # x-component of gradient
F_y = -Y / T_R  # y-component of gradient

# Set forbidden regions to 0
F_x[X < 0] = 0
F_y[X < 0] = 0
F_x[X > 10] = 0
F_y[X > 10] = 0

# Visualize gradient
plt.quiver(X, Y, F_x, F_y)
plt.show()
