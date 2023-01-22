import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2.7 - 3.8

# Read in dataset
dataset = pd.read_csv(f"data/fuzzy/fuzzy5.csv", header=0)

# Add column odd = value from column H if y == "BW" or "SW"
# = value from column A if y == "BL" or "SL"
# = value from column D if y == "D"
dataset['odd'] = np.where(dataset['y'] == "BW", dataset['H'], np.where(dataset['y'] == "SW", dataset['H'], np.where(dataset['y'] == "BL", dataset['A'], np.where(dataset['y'] == "SL", dataset['A'], dataset['D']))))

# Drop all rows where y is D
dataset = dataset[dataset['y'] != "D"]

# Save dataset to file
dataset.to_csv('data/fuzzy/odds.csv', index=False)

# Plot odd distribution on range 1.0 - 6.0
dataset['odd'].hist(bins=1000, range=(1, 10))

# Show plot
plt.show()