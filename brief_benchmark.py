import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

# Create metrcis data
metrics = pd.DataFrame(columns=['H', 'D', 'ROI H', 'ROI D', 'ROI A', 'Bets H', 'Bets D', 'Bets A', 'Profit H', 'Profit D', 'Profit A'])

# Save to file
metrics.to_csv("data/metrics/xgb_brief.csv", index=False)

for h in np.arange(1,10,0.1):
    for a in np.arange(1,10,0.1):
        h = h.round(1)
        a = a.round(1)
        try:
            print(f"Running for H >= {h} and A >= {a}")

            # Load the data
            data = pd.read_csv("data/good/short2.csv")

            print("Length of data: ", len(data))

            data = data[(data['H'] >= h) & (data['A'] >= a)] # -> D, A

            # Drop all rows where H, D, A equal NaN
            data = data.dropna(subset=['H', 'D', 'A'])

            print("Length of data: ", len(data))

            # Create a dictionary to map outcome to integer values
            outcome_map = {'BL': 0, 'SL': 1, 'D': 2, 'SW': 3, 'BW': 4}

            # Create a new column "outcome_num" to store the mapped outcome
            data = data.replace(outcome_map)

            # Define the features and target
            X = data[['H', 'D', 'A', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]
            y = data['Y']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # XGBoost model
            xgb_model = xgb.XGBClassifier()

            # Fit the model with the custom loss function
            xgb_model.fit(X_train, y_train, verbose=True)

            # Save the model
            # xgb_model.save_model("data/models/xgb_brief.sav")
            pickle.dump(xgb_model, open("data/models/xgb_brief.sav", 'wb'))

            xgb_model.fit(X_train, y_train, verbose=True)

            # Make predictions
            y_pred = xgb_model.predict(X_test)

            # Replace BW -> SW, BL -> SL
            y_pred[y_pred == 4] = 3
            y_pred[y_pred == 0] = 1

            # Calculate the accuracy for each outcome
            acc = np.zeros(5)
            for i in range(5):
                acc[i] = np.mean(y_pred[y_test == i] == y_test[y_test == i])
                print(f"Accuracy for outcome {i}: {acc[i]}")

            # Merge the predictions with original data
            data_pred = pd.DataFrame({'Y': y_test, 'Y_pred': y_pred})

            # Replace the Y = 4 and Y = 0 with 3 and 1
            data_pred['Y'][data_pred['Y'] == 4] = 3
            data_pred['Y'][data_pred['Y'] == 0] = 1

            # Merge with H, D, A
            data_pred = data_pred.merge(data[['H', 'D', 'A']], left_index=True, right_index=True)

            base_pred = data_pred

            # Drop all rows where Y_pred == 3
            data_pred = data_pred[data_pred['Y_pred'] == 3]

            bet_h = len(data_pred)
            print("Total bets on H: ", bet_h)

            # Add a column for the profit, set = (H - 1) if Y = 3 and -1 otherwise
            data_pred['Profit'] = np.where(data_pred['Y'] == 3, data_pred['H'] - 1, -1)

            # Calculate the total profit
            profit_h = data_pred['Profit'].sum()
            print("Total profit H: ", profit_h)

            # ROI
            roi_h = (data_pred['Profit'].sum() / len(data_pred)) * 100
            print("ROI H: ", roi_h)

            data_pred = base_pred

            # Drop all rows where Y_pred == 2
            data_pred = data_pred[data_pred['Y_pred'] == 2]

            bet_d = len(data_pred)
            print("Total bets on D: ", bet_d)

            # Add a column for the profit, set = (H - 1) if Y = 3 and -1 otherwise
            data_pred['Profit'] = np.where(data_pred['Y'] == 2, data_pred['D'] - 1, -1)

            # Calculate the total profit
            profit_d = data_pred['Profit'].sum()
            print("Total profit D: ", profit_d)

            # ROI
            roi_d = (data_pred['Profit'].sum() / len(data_pred)) * 100
            print("ROI D: ", roi_d)

            data_pred = base_pred

            # Drop all rows where Y_pred == 1
            data_pred = data_pred[data_pred['Y_pred'] == 1]

            bet_a = len(data_pred)
            print("Total bets on A: ", bet_a)

            # Add a column for the profit, set = (H - 1) if Y = 3 and -1 otherwise
            data_pred['Profit'] = np.where(data_pred['Y'] == 1, data_pred['A'] - 1, -1)

            # Calculate the total profit
            profit_a = data_pred['Profit'].sum()
            print("Total profit A: ", profit_a)

            # ROI
            roi_a = (data_pred['Profit'].sum() / len(data_pred)) * 100
            print("ROI A: ", roi_a)

            # Create row for metrics
            metrics = pd.DataFrame({
                'H': h, 'D': a, 'ROI H': roi_h, 'ROI D': roi_d, 'ROI A': roi_a,
                'Bets H': bet_h, 'Bets D': bet_d, 'Bets A': bet_a,
                'Profit H': profit_h, 'Profit D': profit_d, 'Profit A': profit_a
            }, index=[0])

            # Append to file
            metrics.to_csv("data/metrics/xgb_brief.csv", mode='a', index=False, header=False)
        except:
            print("Error")

        # # Plot the profit
        # data_pred = data_pred.reset_index(drop=True)
        # plt.plot(data_pred['Profit'].cumsum())
        # plt.show()