from keras.layers import BatchNormalization, Dense, Input, Dropout
from keras.models import Model
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
     
def get_data():
    data = pd.read_csv('data/good/fuzzy.csv')
    # Drop D columns
    # data = data.drop(['D'], axis=1)
    # Replace BL, SL, D, SW, BW with 1, 2, 3, 4, 5
    data = data.replace(['BL', 'SL', 'D', 'SW', 'BW'], [1, 2, 3, 4, 5])
    X = data.values[:, 3:-1]
    y = data.values[:, -1]
    y_full = np.zeros((X.shape[0], 6))
    for i, y_i in enumerate(y):
        if y_i == 4 or y_i == 5:
            y_full[i, 0] = 1.0 # win home team
        if y_i == 1 or y_i == 2:
            y_full[i, 1] = 1.0 # win away team
        if y_i == 3:
            y_full[i, 2] = 1.0 # draw
        y_full[i, 3] = X[i, 1] # odds H
        y_full[i, 4] = X[i, 2] # odds D
        y_full[i, 5] = X[i, 3] # odds A

    return X, y_full, y
X, y, outcome = get_data()

X = np.asarray(X).astype(np.float32)


# SPLIT THE DATA IN TRAIN AND TEST DATASET.
train_x, test_x, train_y, test_y, = train_test_split(X,  y)

def odds_loss(y_true, y_pred):
    """
    The function implements the custom loss function mentioned in info.pdf
    
    Inputs
    true : a vector of dimension batch_size, 7. A label encoded version of the output and the backp1_a and backp1_b
    pred : a vector of probabilities of dimension batch_size , 5.
    
    Returns 
    the loss value
    """
    win_home_team = y_true[:, 0:1]
    win_away = y_true[:, 1:2]
    draw = y_true[:, 2:3]
    odds_H = y_true[:, 3:4]
    odds_D = y_true[:, 4:5]
    odds_A = y_true[:, 5:6]
    gain_loss_vector = K.concatenate([win_home_team * (odds_H - 1) + (1 - win_home_team) * -1,
                                      win_away * (odds_A - 1) + (1 - win_away) * -1,
                                      draw * (odds_D - 1) + (1 - draw) * -1,
                                      K.zeros_like(odds_H)], axis=1)
    return -1 * K.mean(K.sum(gain_loss_vector * y_pred, axis=1))
 


# true = K.variable(np.array([[1, 1, 0, 0, 0, 0, 2.0, 3.0]]), dtype='float32')
# pred = K.variable(np.array([[0.6, 0.1, 0.2, 0.05, 0.05, 0.0]]), dtype='float32')

# K.eval(odds_loss(true, pred))

def get_model(input_dim, output_dim, base=1000, multiplier=0.25, p=0.2):
    inputs = Input(shape=(input_dim,))
    l = BatchNormalization()(inputs)
    l = Dropout(p)(l)
    n = base
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(p)(l)
    n = int(n * multiplier)
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(p)(l)
    n = int(n * multiplier)
    l = Dense(n, activation='relu')(l)
    outputs = Dense(output_dim, activation='softmax')(l)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Nadam', loss=odds_loss)
    return model

# Remove 'data/models/odds_loss.hdf5' before running this cell
try:
    os.remove('data/models/odds_loss.hdf5')
except:
    pass

model = get_model(X.shape[1], 4, 1000, 0.9, 0.7)
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
          epochs=200, batch_size=5, callbacks=[EarlyStopping(patience=25),
                                                ModelCheckpoint('data/models/odds_loss.hdf5',
                                                                save_best_only=True)])

print('Training Loss : {}\nValidation Loss : {}'.format(model.evaluate(train_x, train_y), model.evaluate(test_x, test_y)))

# Predict the test set
y_pred = model.predict(test_x)

# Profit
def profit(y_true, y_pred):
    win_home_team = y_true[:, 0:1]
    win_away = y_true[:, 1:2]
    draw = y_true[:, 2:3]
    odds_H = y_true[:, 3:4]
    odds_D = y_true[:, 4:5]
    odds_A = y_true[:, 5:6]
    gain_loss_vector = np.concatenate([win_home_team * (odds_H - 1) + (1 - win_home_team) * -1,
                                      win_away * (odds_A - 1) + (1 - win_away) * -1,
                                      draw * (odds_D - 1) + (1 - draw) * -1,
                                      np.zeros_like(odds_H)], axis=1)
    return np.sum(np.sum(gain_loss_vector * y_pred, axis=1))

print('Profit : {}'.format(profit(test_y, y_pred)))

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()