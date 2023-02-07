import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import pickle

df = pd.read_csv("data/good/ft_df.csv")
# df = df.head(5000)

#dropping columns one wouldn't have before an actual match
cols_to_drop = ['season', 'match_name','date', 'home_team', 'away_team', 'home_score', 'away_score',
                'h_match_points', 'a_match_points']

df.drop( columns = cols_to_drop, inplace = True)

#filling NAs
df.fillna(-33, inplace = True)

#turning the target variable into integers
df['winner'] = np.where(df.winner == 'HOME_TEAM', 2, np.where(df.winner == 'AWAY_TEAM', 1, 0))

#turning categorical into dummy vars
df_dum = pd.get_dummies(df)

np.random.seed(101)

X = df_dum.drop(columns = ['winner'], axis = 1)
y = df_dum.winner.values

test_size = 0.2

# #splitting into train and test set to check which model is the best one to work on
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

# scaling features
scaler = MinMaxScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

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

#getting column names
# featured_columns = pd.DataFrame(rfe.support_,
#                             index = X.columns,
#                             columns=['is_in'])

# featured_columns = featured_columns[featured_columns.is_in == True].index.tolist()
featured_columns = ['h_odd', 'd_odd', 'a_odd', 'ht_rank', 'ht_l_points', 'at_rank', 'at_l_points', 
                    'at_l_wavg_points', 'at_l_wavg_goals', 'at_l_wavg_goals_sf', 'at_win_streak', 
                    'ls_winner_-33', 'ls_winner_HOME_TEAM']

print(featured_columns)

# #column importances for each class
# importances_d = pd.DataFrame(np.exp(rfe.estimator_.coef_[0]),
#                             index = featured_columns,
#                             columns=['coef']).sort_values('coef', ascending = False)

# importances_a = pd.DataFrame(np.exp(rfe.estimator_.coef_[1]),
#                             index = featured_columns,
#                             columns=['coef']).sort_values('coef', ascending = False)

# importances_h = pd.DataFrame(np.exp(rfe.estimator_.coef_[2]),
#                             index = featured_columns,
#                             columns=['coef']).sort_values('coef', ascending = False)

regression = pickle.load(open('data/models/logreg.sav', 'rb'))
tpred_lr = regression.predict(X_test)

# Accuracy
print("Logistic Regression Accuracy: ", accuracy_score(y_test, tpred_lr))

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

test_df['tpred_lr'] = tpred_lr
# Set to tpred_knn if equal to tpred_lr
test_df['winner'] = y_test
test_df['winning_odd'] = test_df.apply(lambda x: get_winning_odd(x), axis = 1)

test_df['lr_profit'] = test_df.winning_odd - 1
test_df.loc[test_df.winner != test_df.tpred_lr, 'lr_profit'] = -1

# Total matches
print('Total Matches: ', len(test_df))

# Drop where winning odd > 3
test_df = test_df[test_df.winning_odd <= 2.8]

# Total bets
print('Total Bets: ', len(test_df))

# Accuracy
print('Logistic Regression Accuracy: ', accuracy_score(test_df.winner, test_df.tpred_lr))

# Profit
print('Logistic Regression Profit: ', test_df.lr_profit.sum())

# Round to 2 decimal places
test_df.lr_profit = test_df.lr_profit.round(2)
test_df.lr_profit = test_df.lr_profit.round(2)


# Save test_df to csv
test_df.to_csv('data/predictions/test_df.csv')

# ROI
print('Logistic Regression ROI: ', test_df.lr_profit.sum()/len(test_df))