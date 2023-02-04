from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Read in dataset
dataset = pd.read_csv(f"data/fuzzy/fuzzy3.csv", header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

# Create a new column "outcome_num" to store the mapped outcome
data = dataset.replace(outcome_map)

X = data.drop(['date','team1','team2','y'], axis=1)
y = data['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define the parameter space for the decision tree
param_grid = {'max_depth': np.arange(1, 11), 'min_samples_split': np.arange(2, 11), 'criterion': ['gini', 'entropy']}

# create an instance of the decision tree classifier
clf = DecisionTreeClassifier()

# use randomized search to optimize the hyperparameters
grid = RandomizedSearchCV(clf, param_grid, cv=5, scoring=make_scorer(accuracy_score))
grid.fit(X_train, y_train)

# print the best hyperparameters
print("Best parameters: ", grid.best_params_)

#print the best score
print("Best score: ", grid.best_score_)
