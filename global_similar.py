import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Read in dataset
dataset = pd.read_csv(f"data/test2.csv", header=0)

print(f'Number of rows: {len(dataset)}')

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
df = dataset.replace(outcome_map)

# Select the columns you want to use for similarity comparisons
data = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]

# Calculate the Euclidean distance between all records
distances = pairwise_distances(data, metric='euclidean')

# Perform K-means clustering with a specified number of clusters
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(distances)

# Add the cluster labels to the original dataframe
df['cluster'] = clusters

# Plot correlation between Y and cluster
plt.scatter(df['Y'], df['cluster'])
plt.xlabel('Outcome')
plt.ylabel('Cluster')
plt.show()