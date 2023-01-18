import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def data(country, learn_period, start_year):
    # Read in dataset
    data = pd.read_csv(f"data/{country}/fuzzy3.csv", header=0)

    # Create a dictionary to map outcome to integer values
    outcome_map = {'D': 0, 'SL': 1, 'BL': 2, 'SW': 3, 'BW': 4}

    # Create a new column "outcome_num" to store the mapped outcome
    data = data.replace(outcome_map)

    # Get only data where date is contained in start_year + learn_period
    original_data = data
    data = data[data['date'].str.contains(f"{start_year}")]
    for i in range(learn_period):
        new_data = original_data[original_data['date'].str.contains(f"{start_year + i}")]
        data = pd.concat([data, new_data])

    test_data = data

    # Assign the input variables to X and the output variable to y
    X = data.drop(['date','team1','team2','y'], axis=1)
    y = data['y']

    # split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    return X_train, X_test, y_train, y_test, test_data

def plot_mean(name, country, start_year, learn_period):
    # Plot accuracies.csv
    accuracies = pd.read_csv(f"data/accuracies_{name}.csv", header=None)
    accuracies.columns = ['D', 'SL', 'BL', 'SW', 'BW', 'Overall']

    # Plot mean accuracy
    accuracies.mean().plot(kind='bar')
    plt.xlabel('Outcome')
    plt.ylabel('Accuracy')
    plt.savefig(f"data/{country}_{name}_{start_year}_{learn_period}_.png")
    plt.close()

    # round to 2 decimal places
    values = np.round(accuracies.loc[0].values, 2)

    # Join values with commas
    values = ",".join(map(str, values))

    # Append mean accuracy to mean.csv
    with open(f"data/mean/{name}.csv", 'a') as f:
        # For each accuracy, round to 2 decimal places
        f.write(f"{country},{start_year},{learn_period},{values}\n")

def process(y_test, y_pred, name, test_data):
    # Convert predictions to integer values
    y_pred = np.round(y_pred).astype(int)

    # create a dataframe with test and prediction results
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

    # Create a new column "correct" to store whether the prediction was correct
    df['correct'] = df['y_test'] == df['y_pred']

    # count the number of correct predictions of 'D' outcome
    df_d = df[df['y_pred'] == 0]
    correct_d = df_d[df_d['correct'] == True]
    if len(df_d) == 0:
        acc_d = ""
    else:
        acc_d = len(correct_d) / len(df_d)
    print("Accuracy for D outcome: ", acc_d)

    # count the number of correct predictions of 'SL' outcome
    df_sl = df[df['y_pred'] == 1]
    correct_sl = df_sl[df_sl['correct'] == True]
    if len(df_sl) == 0:
        acc_sl = ""
    else:
        acc_sl = len(correct_sl) / len(df_sl)
    print("Accuracy for SL outcome: ", acc_sl)

    # count the number of correct predictions of 'BL' outcome
    df_bl = df[df['y_pred'] == 2]
    correct_bl = df_bl[df_bl['correct'] == True]
    if len(df_bl) == 0:
        acc_bl = ""
    else:
        acc_bl = len(correct_bl) / len(df_bl)
    print("Accuracy for BL outcome: ", acc_bl)

    # count the number of correct predictions of 'SW' outcome
    df_sw = df[df['y_pred'] == 3]
    correct_sw = df_sw[df_sw['correct'] == True]
    if len(df_sw) == 0:
        acc_sw = ""
    else:
        acc_sw = len(correct_sw) / len(df_sw)
    print("Accuracy for SW outcome: ", acc_sw)

    # count the number of correct predictions of 'BW' outcome
    df_bw = df[df['y_pred'] == 4]
    correct_bw = df_bw[df_bw['correct'] == True]
    if len(df_bw) == 0:
        acc_bw = ""
    else:
        acc_bw = len(correct_bw) / len(df_bw)
    print("Accuracy for BW outcome: ", acc_bw)

    # Calculate overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Overall accuracy: ", acc)

    # Save all accuracies to a file
    with open(f"data/accuracies_{name}.csv", 'a') as f:
        data = [acc_d, acc_sl, acc_bl, acc_sw, acc_bw, acc]
        # Convert list to string
        data = ','.join(map(str, data)) + "\n"
        f.write(data)

    accuracies = pd.read_csv(f"data/accuracies_{name}.csv", header=None)
    accuracies.columns = ['D', 'SL', 'BL', 'SW', 'BW', 'Overall']

    # Plot all accuracies
    accuracies.plot()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    # plt.savefig(f"data/accuracies_{name}.png")

    plt.close()