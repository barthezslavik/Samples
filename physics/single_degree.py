import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force

# Set up grid
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 30))

# Create a figure and axes object
fig, ax = plt.subplots()

# Initialize a quiver plot to display the gradient
Q = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y))

# Function to update the quiver plot at each frame
def update(i):
    # Calculate gradient of causal entropic force
    F_x = -X / T_R * np.cos(i * np.pi / 10)  # x-component of gradient
    F_y = -Y / T_R * np.sin(i * np.pi / 10)  # y-component of gradient

    # Update the quiver plot
    Q.set_UVC(F_x, F_y)

# Create an animation object
ani = animation.FuncAnimation(fig, update, frames=100, interval=100)

# Show the plot
plt.show()
