import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Define the input and output variables
outcomes = np.array([0, 1, 2, 3, 4])
scores = np.array([-5, -3, 0, 2, 5])

# Define the membership functions for the input variables
BL_mem_func = np.array([1, 0, 0, 0, 0])
SL_mem_func = np.array([0, 1, 0, 0, 0])
D_mem_func = np.array([0, 0, 1, 0, 0])
SW_mem_func = np.array([0, 0, 0, 1, 0])
BW_mem_func = np.array([0, 0, 0, 0, 1])

# Train neural network to approximate membership functions
nn_outcomes = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_outcomes.fit(outcomes.reshape(-1, 1), np.column_stack((BL_mem_func, SL_mem_func, D_mem_func, SW_mem_func, BW_mem_func)))
nn_scores = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_scores.fit(scores.reshape(-1, 1), np.column_stack((BL_mem_func, SL_mem_func, D_mem_func, SW_mem_func, BW_mem_func)))

# Use the neural network to get the membership values for a smoother membership function
X_smooth = np.linspace(-5,5, 200)
outcomes_smooth = nn_outcomes.predict(X_smooth.reshape(-1,1))
scores_smooth = nn_scores.predict(X_smooth.reshape(-1,1))

# Plot the original membership function
plt.plot(outcomes, BL_mem_func, 'o', label='BL')
plt.plot(outcomes, SL_mem_func, 'o', label='SL')
plt.plot(outcomes, D_mem_func, 'o', label='D')
plt.plot(outcomes, SW_mem_func, 'o', label='SW')
plt.plot(outcomes, BW_mem_func, 'o', label='BW')

# Plot the smoothed membership function
plt.plot(X_smooth, outcomes_smooth[:,0], label='BL_smooth')
plt.plot(X_smooth, outcomes_smooth[:,1], label='SL_smooth')
plt.plot(X_smooth, outcomes_smooth[:,2], label='D_smooth')
plt.plot(X_smooth, outcomes_smooth[:,3], label='SW_smooth')
plt.plot(X_smooth, outcomes_smooth[:,4], label='BW_smooth')

plt.legend()
plt.xlabel('Outcomes')
plt.ylabel('Membership')
plt.show()
