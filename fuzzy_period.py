import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Read in dataset
data_train = pd.read_csv('data/fuzzy/train.csv', header=0)
data_test = pd.read_csv('data/fuzzy/test.csv', header=0)

# Create a dictionary to map outcome to integer values
outcome_map = {'SW': 0, 'SL': 1, 'D': 2, 'BW': 3, 'BL': 4}

# Create a new column "outcome_num" to store the mapped outcome
data_train = data_train.replace(outcome_map)
data_test = data_test.replace(outcome_map)

X_train = data_train.drop(['date','team1','team2','y'], axis=1)
y_train = data_train['y']
X_test = data_test.drop(['date','team1','team2','y'], axis=1)
y_test = data_test['y']

# Train the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Convert predictions to integer values
y_pred = np.round(y_pred).astype(int)

# create a dataframe with test and prediction results
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

# count the number of correct predictions of 'SW' outcome
df['correct'] = df['y_test'] == df['y_pred']
df_sw = df[df['y_pred'] == 0]
correct_sw = df_sw[df_sw['correct'] == True]
acc_sw = "" if len(df_sw) == 0 else len(correct_sw) / len(df_sw)
print("Accuracy for SW outcome: ", acc_sw)

# count the number of correct predictions of 'SL' outcome
df_sl = df[df['y_pred'] == 1]
correct_sl = df_sl[df_sl['correct'] == True]
acc_sl = "" if len(df_sl) == 0 else len(correct_sl) / len(df_sl)
print("Accuracy for SL outcome: ", acc_sl)

# count the number of correct predictions of 'D' outcome
df_d = df[df['y_pred'] == 2]
correct_d = df_d[df_d['correct'] == True]
acc_d = "" if len(df_d) == 0 else len(correct_d) / len(df_d)
print("Accuracy for D outcome: ", acc_d)

# count the number of correct predictions of 'BW' outcome
df_bw = df[df['y_pred'] == 3]
correct_bw = df_bw[df_bw['correct'] == True]
acc_bw = "" if len(df_bw) == 0 else len(correct_bw) / len(df_bw)
print("Accuracy for BW outcome: ", acc_bw)

# count the number of correct predictions of 'BL' outcome
df_bl = df[df['y_pred'] == 4]
correct_bl = df_bl[df_bl['correct'] == True]
acc_bl = "" if len(df_bl) == 0 else len(correct_bl) / len(df_bl)
print("Accuracy for BL outcome: ", acc_bl)

# Calculate overall accuracy
acc = accuracy_score(y_test, y_pred)
# print("Overall accuracy: ", acc)

print("")

# Coefficients
print(f"Coefficient for SW > {round(100/(acc_sw * 100), 2)}" if acc_sw != 0 else "")
print(f"Coefficient for SL > {round(100/(acc_sl * 100), 2)}" if acc_sl != 0 else "")
print(f"Coefficient for D > {round(100/(acc_d * 100), 2)}" if acc_d != 0 else "")
print(f"Coefficient for BW > {round(100/(acc_bw * 100), 2)}" if acc_bw != 0 else "")
print(f"Coefficient for BL > {round(100/(acc_bl * 100), 2)}" if acc_bl != 0 else "")

# Save all accuracies to a file
with open('data/accuracies_all.csv', 'a') as f:
    data = [acc_sw, acc_sl, acc_d, acc_bw, acc_bl, acc]
    # Convert list to string
    data = ','.join(map(str, data)) + "\n"
    f.write(data)

# Plot accuracies.csv
accuracies = pd.read_csv('data/accuracies_all.csv', header=None)
accuracies.columns = ['SW', 'SL', 'D', 'BW', 'BL', 'Overall']

# Plot mean accuracy
accuracies.mean().plot(kind='bar')
plt.xlabel('Outcome')
plt.ylabel('Accuracy')
plt.savefig('data/mean_accuracy_xg.png')

# Plot all accuracies
accuracies.plot()
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.savefig('data/accuracies_xg.png')