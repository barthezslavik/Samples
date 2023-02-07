import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

limit = 1000
test_size = 0.5

df = pd.read_csv("data/good/ft_df.csv")
df = df.head(limit); print("Limit: ", limit)

#dropping columns one wouldn't have before an actual match
cols_to_drop = ['season', 'match_name','date', 'home_team', 'away_team', 
                'home_score', 'away_score',
                'h_match_points', 'a_match_points']

df.drop( columns = cols_to_drop, inplace = True)

# Drop NAs rows if h_odd, d_odd, a_odd are NAs
df.dropna(subset=['h_odd', 'd_odd', 'a_odd'], inplace=True)

#filling NAs
df.fillna(-33, inplace = True)

#turning the target variable into integers
df['winner'] = np.where(df.winner == 'HOME_TEAM', 2, np.where(df.winner == 'AWAY_TEAM', 1, 0))

# Save df to csv
# df.to_csv("data/good/ft_winner.csv", index=False)

#turning categorical into dummy vars
df_dum = pd.get_dummies(df)

X = df_dum.drop(columns = ['winner'], axis = 1)
y = df_dum.winner.values

#scaling features
scaler = MinMaxScaler()

#best classifier on training data
clf = LogisticRegression(max_iter = 1000, multi_class = 'multinomial')

#getting the best 13 features from RFE
rfe = RFE(estimator = clf, n_features_to_select = 13, step=1)
rfe.fit(X, y)
X_transformed = rfe.transform(X)

np.random.seed(101)
X_train, X_test, y_train, y_test = train_test_split(X_transformed,y, test_size = test_size)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

featured_columns = ['h_odd', 'd_odd', 'a_odd', 'ht_rank', 'ht_l_points', 'at_rank', 'at_l_points', 
                    'at_l_wavg_points', 'at_l_wavg_goals', 'at_l_wavg_goals_sf', 'at_win_streak', 
                    'ls_winner_-33', 'ls_winner_HOME_TEAM']

#tuning logistic regression
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
 'fit_intercept': (True, False), 'solver' : ('newton-cg', 'sag', 'saga', 'lbfgs'), 'class_weight' : (None, 'balanced')}

gs = GridSearchCV(clf, parameters, scoring='accuracy', cv=3)
gs.fit(X_train,y_train)

#testing models on unseen data 
prediction = gs.best_estimator_.predict(X_test)

# Accuracy
print("Logistic Regression Accuracy: ", accuracy_score(y_test, prediction))

# Accuracy for each prediction
for i in range(3):
    print("Accuracy for prediction {}: {}".format(i, accuracy_score(y_test[y_test == i], prediction[y_test == i])))

#function to get winning odd value in simulation dataset
def get_winning_odd(df):
    udi = 0
    if df.winner == 2:
        result = df.h_odd + udi
    elif df.winner == 1:
        result = df.a_odd + udi
    else:
        result = df.d_odd
    return result

#creating dataframe with test data to simulate betting winnings with models
test_df = pd.DataFrame(scaler.inverse_transform(X_test),columns =  featured_columns)

test_df['prediction'] = prediction
# Set to tpred_knn if equal to prediction
test_df['winner'] = y_test

# Append h_odd if not in featured_columns
# if 'h_odd' not in featured_columns:
#     # Merge h_odd to test_df
#     test_df = test_df.merge(df[['h_odd']], left_index=True, right_index=True)
# if 'a_odd' not in featured_columns:
#     # Merge a_odd to test_df
#     test_df = test_df.merge(df[['a_odd']], left_index=True, right_index=True)
# if 'd_odd' not in featured_columns:
#     # Merge d_odd to test_df
#     test_df = test_df.merge(df[['d_odd']], left_index=True, right_index=True)

# print(test_df.head(10))

# Plot distribution of h_odd and prediction == 2 and winner == 2 # < 4.7
# test_df[(test_df.prediction == 2) & (test_df.winner == 2)].h_odd.hist()

# Plot distribution of a_odd and prediction == 1 and winner == 1
# test_df[(test_df.prediction == 1) & (test_df.winner == 1)].a_odd.hist() # < 12.5

# Plot distribution of d_odd and prediction == 0 and winner == 0
# test_df[(test_df.prediction == 0) & (test_df.winner == 0)].d_odd.hist() # < 3.45

# Drop rows if prediction == 2 and h_odd > 4.7    
# test_df.drop(test_df[(test_df.prediction == 2) & (test_df.h_odd > 4.7)].index, inplace=True)

test_df['winning_odd'] = test_df.apply(lambda x: get_winning_odd(x), axis = 1)

test_df['profit'] = test_df.winning_odd - 1
test_df.loc[test_df.winner != test_df.prediction, 'profit'] = -1

# ROI
print('Logistic Regression ROI: ', test_df.profit.sum()/len(test_df))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(test_df.profit.cumsum(), label='Logistic Regression')

# Add regression line
# plt.plot(np.poly1d(np.polyfit(range(len(test_df)), test_df.profit.cumsum(), 1))(range(len(test_df))), label='Logistic Regression Regression Line')
plt.legend()
plt.title('Profit Curve')
plt.xlabel('Number of bets')
plt.ylabel('Profit')
# plt.show()