import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 1  # length of pendulum
G = 9.81  # acceleration due to gravity
MASS = 1  # mass of pendulum
TAU = 10  # causal entropic force parameter
T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force

# Generate grid of points to evaluate gradient at
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)

# Calculate gradient of causal entropic force
s = -(X**2 + Y**2).sum() / (2 * T_R)  # entropy of system
grad_s_x, grad_s_y = -X / T_R, -Y / T_R  # gradient of entropy
F_x = T_R * grad_s_x / TAU
F_y = T_R * grad_s_y / TAU

# Plot gradient of causal entropic force
plt.quiver(X, Y, F_x, F_y)
plt.show()
