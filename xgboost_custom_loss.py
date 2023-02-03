import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read the data
train = pd.read_csv("data/good/discovery14.csv")

# Drop NA values
train = train.dropna()

# Replace Div with numbers
replace_map = {'E0':0, 'SP1':1, 'I1':2, 'F1':3, 'D1':4, 'N1':5, 'P1':6, 'SC0':7, 'B1':8, 'G1':9, 
'T1':10, 'F2':11, 'I2':12, 'E1':13, 'D2':14, 'N2':15, 'P2':16, 'SC1':17, 'B2':18, 'G2':19, 'T2':20,
'SC2': 21, 'SC3': 22, 'E2': 23, 'E3': 24, 'EC': 25, 'SP2': 26}

train = train.replace(replace_map)

# Add column FTR = 2 if HomeTeam wins, 1 if draw, 0 if AwayTeam wins
train['FTR'] = 2
train.loc[train['FTHG'] < train['FTAG'], 'FTR'] = 0
train.loc[train['FTHG'] == train['FTAG'], 'FTR'] = 1

# Drop unnecessary columns
X = train.drop(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'], axis=1)
print(X)
y = train['FTR']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

# Use neural network
from sklearn.neural_network import MLPClassifier

# Create a neural network
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

# Train the model
mlp.fit(X_train, y_train)

# Predict the outcome
y_pred = mlp.predict(X_test)

print("Neural network model")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

print(accuracy)

# Xgboost model

import xgboost as xgb

xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print("XGBoost model")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {}".format(accuracy))

# Deep learning model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(X.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, verbose=0)

y_pred = model.predict(X_test)

print("Deep learning model")

# Accuracy score
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print("Accuracy: {}".format(accuracy))
