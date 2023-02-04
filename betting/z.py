import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/Germany_93_21/fuzzy3.csv', header=0)

# Prepare the dataset
# Example assumes that the dataset is stored in a variable called 'data'
X = data.drop(columns=["y", "date", "team1", "team2", "x6", "x7", "x8", "x9", "x10", "x11", "x12"])
y = data["y"]

# Convert string features to numerical values using LabelEncoder
encoder = LabelEncoder()
for col in X.columns:
    X[col] = encoder.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# convert string labels to numerical values using LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Define the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

def get_z(results, model, encoder):
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

    original_results = results

    # Convert ('BL', 'BL', 'BW', 'BL', 'BL') to [[0, 0, 1, 0, 0]]
    nn_results = []
    for result in results:
        nn_results.append(encoder.transform([result]))
    nn_results = np.array(nn_results)
    nn_results = nn_results.reshape(1, -1)

    # Predict the Z value
    nn_z_values = model.predict(nn_results)
    nn_z_values = np.argmax(nn_z_values, axis=1)
    nn_z_values = encoder.inverse_transform(nn_z_values)

    # Check if the Z value is in the balanced dictionary
    if original_results in z_values_balanced:
        return z_values_balanced[original_results]
    else:
        return nn_z_values[0]

outcomes = ["SW", "SL", "D", "BW", "BL"]
for i in range(100):
    results = random.choices(outcomes, k=5)
    outcome = get_z(tuple(results), model, encoder)
    print(outcome)