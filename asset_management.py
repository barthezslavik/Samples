import numpy as np
import pandas as pd
import scipy.optimize as optimize

# Load data for a set of assets
data = pd.read_csv('data/asset_data.csv')

# Calculate the mean returns and covariance matrix for the assets
mean_returns = data.mean()
cov_matrix = data.cov()

# Define a function to minimize the negative of the portfolio entropy
def portfolio_entropy(x, mean_returns, cov_matrix):
    return -np.sum(x * np.log(x))

# Define constraints for the optimization
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # sum of weights must be 1
    {'type': 'ineq', 'fun': lambda x: np.dot(x, mean_returns) - 0.1},  # expected return must be at least 0.1
]

# Set bounds on the weights
bounds = [(0, 1)] * len(mean_returns)

# Initialize the optimization starting point
x0 = np.ones(len(mean_returns)) / len(mean_returns)

# Run the optimization to find the maximum entropy portfolio
result = optimize.minimize(portfolio_entropy, x0, args=(mean_returns, cov_matrix),
                           constraints=constraints, bounds=bounds, method='SLSQP')

# Print the optimal weights
print(result.x)
