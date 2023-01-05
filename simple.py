import numpy as np

# Constants
MASS = 1  # mass of particle (kg)
K = 1  # spring constant (N/m)
L = 1  # width of box (m)
TAU = 1  # time scale (s)
T_R = 300  # temperature of reservoir (K)
TIMESTEP = 0.01  # time step for simulation (s)

# Initialize position and velocity
x = 0.5  # initial position (m)
v = 0.0  # initial velocity (m/s)

# Main loop
for t in range(10000):
    # Calculate energetic force
    energetic_force = -K * x

    # Calculate random force
    random_force = np.random.normal(0, np.sqrt(MASS * T_R * TIMESTEP))

    # Calculate causal entropic force
    s = -(x**2 + v**2) / (2 * T_R)  # entropy of system
    grad_s = np.array([-x, -v])  # gradient of entropy
    causal_entropic_force = T_R * grad_s / TAU

    # Calculate total force
    total_force = energetic_force + random_force + causal_entropic_force

    # print(total_force)
    # Update velocity
    v += total_force# / MASS * TIMESTEP

    # Update position
    x += v * TIMESTEP

    # Check for collision with walls
    # if (x < 0) or (x > L):
    #    v *= -1

    # Print position and velocity
    print(f"t = {t * TIMESTEP:.2f}, x = {x:.2f}, v = {v:.2f}")