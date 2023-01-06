import numpy as np

# Constants
MASS = 100  # particle mass
T_R = 4e5  # temperature of heat reservoir
T_C = 5 * T_R  # scaling factor for magnitude of causal entropic force
TAU = 10  # causal entropic force parameter
TIMESTEP = 0.025  # time step for simulation

# Initialize position and momentum
position = np.array([0, 0], dtype=np.float64)
momentum = np.array([0, 0], dtype=np.float64)

# Main loop
for t in range(1000):
    # print("===============================================", t)
    # Calculate causal entropic force
    s = -(position**2 + momentum**2) / (2 * T_R)  # entropy of system
    grad_s = np.array([-position, -momentum])  # gradient of entropy
    causal_entropic_force = T_R * grad_s / TAU
    # print("causal_entropic_force", causal_entropic_force)

    # Calculate random force
    random_force = np.random.normal(0, np.sqrt(2 * MASS * T_R * TIMESTEP), size=2)
    # print("random_force", random_force)

    # Calculate total force
    total_force = causal_entropic_force[0] + random_force[0]

    # print("momentum", momentum, "mass", MASS, "timestamp", TIMESTEP, "total_force", total_force[0])
    # Update position and momentum
    position += (momentum / MASS) * TIMESTEP + (total_force / MASS) * TIMESTEP**2 / 2
    momentum += total_force * TIMESTEP
    print("position", position, "momentum", momentum)
