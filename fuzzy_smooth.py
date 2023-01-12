import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Define the input and output variables
temperature = np.linspace(-20, 41, 1000)
humidity = np.linspace(0, 101, 1000)
comfort = np.zeros_like(temperature)

# Define the membership functions for the input variables
temp_cold_mem_func = np.clip(1 - abs(temperature + 20)/20, 0, 1)
temp_cool_mem_func = np.clip(1 - abs(temperature)/20, 0, 1)
temp_warm_mem_func = np.clip(1 - abs(temperature - 20)/20, 0, 1)
humidity_dry_mem_func = np.clip(1 - abs(humidity - 50)/50, 0, 1)
humidity_comfortable_mem_func = np.clip(1 - abs(humidity - 75)/25, 0, 1)

# Train neural network to approximate membership functions
nn_temp = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_temp.fit(temperature.reshape(-1, 1), np.column_stack((temp_cold_mem_func, temp_cool_mem_func, temp_warm_mem_func)))
nn_humidity = MLPRegressor(hidden_layer_sizes=(20, 20))
nn_humidity.fit(humidity.reshape(-1, 1), np.column_stack((humidity_dry_mem_func, humidity_comfortable_mem_func)))

# Use the neural network to get the membership values for a smoother membership function
X_smooth = np.linspace(-20,41, 200)
temp_smooth = nn_temp.predict(X_smooth.reshape(-1,1))
hum_smooth = nn_humidity.predict(X_smooth.reshape(-1,1))

# Plot the original and the smoothed membership function
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(temperature, temp_cold_mem_func, 'b', linewidth=1.5, label='original')
plt.plot(X_smooth, temp_smooth[:,0], 'r', linewidth=1.5, label='smooth')
plt.legend()
plt.subplot(1,2,2)
plt.plot(humidity, humidity_dry_mem_func, 'b', linewidth=1.5, label='original')
plt.plot(X_smooth, hum_smooth[:,0], 'r', linewidth=1.5, label='smooth')
plt.legend()
plt.show()

# Define the fuzzy rules
rule1 = np.fmin(temp_smooth[:,0], hum_smooth[:,0])
rule2 = np.fmin(temp_smooth[:,1], hum_smooth[:,1])
rule3 = np.fmin(temp_smooth[:,2], hum_smooth[:,0])

# Defuzzification 
comfort = np.fmax(rule1, np.fmax(rule2, rule3))

# Plot defuzzified output
defuzzified_value = np.mean(X_smooth[comfort == np.max(comfort)])
print('Defuzzified value: ', defuzzified_value)

# plt.figure()
# plt.plot(X_smooth, comfort)
# plt.axvline(defuzzified_value, color='r', linestyle='--')
# plt.show()