import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import glob
import random

def data():
    # Read in dataset
    # data_train = pd.read_csv('data/fuzzy/train.csv', header=0)
    # data_test = pd.read_csv('data/fuzzy/test.csv', header=0)

    # Get random dataset
    folders = glob.glob('data/*')
    folder = random.choice(folders)
    print("==================================")
    print(folder)
    print("==================================")

    # Split fuzzy3.csv into train and test
    data = pd.read_csv(folder + '/fuzzy3.csv', header=0)
    n = int(0.8 * len(data))
    data_train = data.head(n)
    print(data_train.tail())
    
    data_test = data.tail(len(data) - n)
    print(data_test.head())

    # Create a dictionary to map outcome to integer values
    outcome_map = {'SW': 0, 'SL': 1, 'D': 2, 'BW': 3, 'BL': 4}

    # Create a new column "outcome_num" to store the mapped outcome
    data_train = data_train.replace(outcome_map)
    data_test = data_test.replace(outcome_map)

    X_train = data_train.drop(['date','team1','team2','y'], axis=1)
    y_train = data_train['y']
    X_test = data_test.drop(['date','team1','team2','y'], axis=1)
    y_test = data_test['y']

    return X_train, X_test, y_train, y_test, data_test

def process(y_test, y_pred, data_test, name):
    # Convert predictions to integer values
    y_pred = np.round(y_pred).astype(int)

    # create a dataframe with test and prediction results
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

    # count the number of correct predictions of 'SW' outcome
    df['correct'] = df['y_test'] == df['y_pred']
    df_sw = df[df['y_pred'] == 0]
    correct_sw = df_sw[df_sw['correct'] == True]
    acc_sw = len(correct_sw) / len(df_sw)
    print("Accuracy for SW outcome: ", acc_sw)

    # count the number of correct predictions of 'SL' outcome
    df_sl = df[df['y_pred'] == 1]
    correct_sl = df_sl[df_sl['correct'] == True]
    if len(df_sl) == 0:
        acc_sl = ""
    else:
        acc_sl = len(correct_sl) / len(df_sl)
    print("Accuracy for SL outcome: ", acc_sl)

    # count the number of correct predictions of 'D' outcome
    df_d = df[df['y_pred'] == 2]
    correct_d = df_d[df_d['correct'] == True]
    acc_d = len(correct_d) / len(df_d)
    print("Accuracy for D outcome: ", acc_d)

    # count the number of correct predictions of 'BW' outcome
    df_bw = df[df['y_pred'] == 3]
    correct_bw = df_bw[df_bw['correct'] == True]
    if len(df_bw) == 0:
        acc_bw = ""
    else:
        acc_bw = len(correct_bw) / len(df_bw)
    print("Accuracy for BW outcome: ", acc_bw)

    # count the number of correct predictions of 'BL' outcome
    df_bl = df[df['y_pred'] == 4]
    correct_bl = df_bl[df_bl['correct'] == True]
    if len(df_bl) == 0:
        acc_bl = ""
    else:
        acc_bl = len(correct_bl) / len(df_bl)
    print("Accuracy for BL outcome: ", acc_bl)

    # Calculate overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Overall accuracy: ", acc)

    # Save all accuracies to a file
    with open(f"data/accuracies_{name}.csv", 'a') as f:
        data = [acc_sw, acc_sl, acc_d, acc_bw, acc_bl, acc]
        # Convert list to string
        data = ','.join(map(str, data)) + "\n"
        f.write(data)

    # Plot accuracies.csv
    accuracies = pd.read_csv(f"data/accuracies_{name}.csv", header=None)
    accuracies.columns = ['SW', 'SL', 'D', 'BW', 'BL', 'Overall']

    # Plot mean accuracy
    accuracies.mean().plot(kind='bar')
    plt.xlabel('Outcome')
    plt.ylabel('Accuracy')
    plt.savefig(f"data/mean_accuracy_{name}.png")

    # Plot all accuracies
    accuracies.plot()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig(f"data/accuracies_{name}.png")

    # restore 0 -> SW, 1 -> SL, 2 -> D, 3 -> BW, 4 -> BL
    # y_test = y_test.replace({0: 'SW', 1: 'SL', 2: 'D', 3: 'BW', 4: 'BL'})
    # y_pred = pd.Series(y_pred).replace({0: 'SW', 1: 'SL', 2: 'D', 3: 'BW', 4: 'BL'})

    # Merge team1, team2, test and prediction results
    # date = pd.Series(data_test['date'])
    # team1 = pd.Series(data_test['team1'])
    # team2 = pd.Series(data_test['team2'])
    # correct = pd.Series(y_test == y_pred)
    # prediction = pd.concat([date, team1, team2, y_test, y_pred, correct], axis=1)
    # prediction.columns = ['Date', 'Home', 'Away', 'Result', 'Prediction', 'Correct']

    # Drop all BW and BL predictions
    # prediction = prediction[prediction['Prediction'] != 'BW']
    # prediction = prediction[prediction['Prediction'] != 'BL']
    # prediction.to_csv('data/prediction.csv', index=False)