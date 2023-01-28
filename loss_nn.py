import numpy as np
import pandas as pd
import tensorflow as tf

# Load the dataset
df = pd.read_csv("data/good/train.csv")

# Define the features and labels
X = df[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]]
y = df["Y"]

# Define the odds
odds = df[["H", "D", "A"]]

# Define the custom loss function
def custom_loss(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    bet_returns = tf.gather_nd(odds.values, tf.stack((tf.range(tf.shape(y_pred)[0]), y_pred), axis=1))
    profit = tf.reduce_sum(bet_returns)
    return -profit

# Create a neural network model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_dim=12, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# Compile the model with the custom loss function
model.compile(loss=custom_loss, optimizer='adam')

print(model.summary())

# Fit the model to the data
# model.fit(X, y, epochs=100, batch_size=32)