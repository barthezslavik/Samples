import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

z_values_balanced = {
    ("BL", "BL", "BL", "BL", "BL"): "BL",
    ("BW", "SL", "BL", "SL", "BW"): "BL",
    ("SW", "BL", "SL", "SL", "SW"): "BL",
    ("SL", "SL", "SL", "SL", "SL"): "SL",
    ("D", "SL", "SL", "D", "D"): "SL",
    ("SW", "D", "SL", "SL", "SW"): "SL",
    ("D", "D", "D", "D", "D"): "D",
    ("SL", "SW", "SW", "D", "SL"): "D",
    ("D", "D", "SW", "SW", "D"): "D",
    ("SW", "SW", "SW", "SW", "SW"): "SW",
    ("D", "BW", "BW", "SW", "D"): "SW",
    ("SL", "SW", "SW", "BW", "SL"): "SW",
    ("BW", "BW", "BW", "BW", "BW"): "BW",
    ("SL", "BW", "SW", "BW", "SL"): "BW",
    ("BL", "SW", "BW", "SW", "BL"): "BW",
}

fuzzy_sets = {}
for key, value in z_values_balanced.items():
    fuzzy_sets[value] = fuzzy_sets.get(value, [])
    fuzzy_sets[value].append(key)

# Create universe variables
x_universe = np.linspace(0, 1, 100)
y_universe = np.linspace(0, 1, 100)
z_universe = np.linspace(0, 1, 100)
w_universe = np.linspace(0, 1, 100)
v_universe = np.linspace(0, 1, 100)

fuzzy_sets = {}
for key, value in z_values_balanced.items():
    fuzzy_sets[value] = fuzzy_sets.get(value, [])
    fuzzy_sets[value].append(fuzz.trimf(v_universe, [0, 0, 1]))

for value, fuzzy_var in fuzzy_sets.items():
    for i, fuzzy_set in enumerate(fuzzy_var):
        plt.plot(v_universe, fuzzy_set, label=f"{value}_{i}")
    
plt.legend()
plt.title(value)
plt.show()
