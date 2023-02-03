import numpy as np
import matplotlib.pyplot as plt

def ising_model(T, L):
    np.random.seed(1234)
    # Initialize the spin grid with random spins
    spin_grid = np.random.choice([-1, 1], size=(L, L))
    # Define the energy change for flipping a spin
    def deltaE(S, i, j):
        return 2 * S[i, j] * (S[i, (j + 1) % L] + S[i, (j - 1 + L) % L] + S[(i + 1) % L, j] + S[(i - 1 + L) % L, j])
    # Monte Carlo simulation
    for t in range(100000):
        i, j = np.random.randint(0, L, size=2)
        dE = deltaE(spin_grid, i, j)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spin_grid[i, j] = -spin_grid[i, j]
    return spin_grid

# Plot the results for different temperatures
plt.figure(figsize=(10, 8))
for i, T in enumerate([1, 5, 10]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(ising_model(T, 20), cmap='gray', vmin=-1, vmax=1)
    plt.title("T = {}".format(T))
plt.show()
