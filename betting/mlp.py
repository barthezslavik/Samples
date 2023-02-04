from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

# Read the data from the CSV file into a pandas DataFrame
df = pd.read_csv('data/mlp.csv')

# Split the data into input and output columns
X = df[['input1', 'input2']]
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network
nn = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', 
                  alpha=0.001, batch_size='auto', learning_rate='constant', 
                  learning_rate_init=0.01, power_t=0.5, max_iter=1000, 
                  shuffle=True, random_state=None, tol=0.0001, 
                  verbose=False, warm_start=False, momentum=0.9, 
                  nesterovs_momentum=True, early_stopping=False, 
                  validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
                  epsilon=1e-08, n_iter_no_change=10)

# Train the neural network
nn.fit(X_train, y_train)

# Make predictions on the test set
predictions = nn.predict(X_test)

# Evaluate the performance of the predictions
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, predictions))
