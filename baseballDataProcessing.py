# baseballDataProcessing.py

import pandas as pd
import numpy as np

batting_df = pd.read_csv('./data/Batting.csv')

#display lots of columns
pd.set_option('display.width', 5000) 
pd.set_option('display.max_columns', 40)

# remove statistics which didn't get recorded until sometime in 1900s
batting_df.drop(batting_df.columns[17:], axis=1, inplace=True)

# fill NaN values with 0
batting_df.fillna(0, inplace=True)

def score(row):
    return row['R'] + row['H'] + row['2B'] + (2 * row['3B']) + (3 * row['HR']) + row['RBI'] + row['SB']

batting_df['score'] = batting_df.apply(lambda row: score(row), axis=1)


def create_player_year_id(df):
    return df['playerID'] + "_" + str(df['yearID'])

batting_index = batting_df.apply(create_player_year_id, axis=1)
batting_df.set_index(batting_index, inplace=True)

agg_batting_df = batting_df.groupby(batting_df.index).agg('sum')
agg_batting_df.drop(['yearID', 'stint'], axis=1, inplace=True)
agg_batting_df['playerID'] = agg_batting_df.index
agg_batting_df['playerID'] = agg_batting_df['playerID'].apply(lambda x: x[:-5])

master_df = pd.read_csv('./data/Master.csv')
master_df = master_df[['playerID', 'birthYear', 'nameFirst', 'nameLast', 'weight', 'height']]

# merge birthyear, height, weight, etc info 
batters2 = pd.merge(left=batting_df, right=master_df, how='left', left_on='playerID', right_on='playerID')

batting_index = batters2.apply(create_player_year_id, axis=1)
batters2.set_index(batting_index, inplace=True)
# index by playerID and year since each player is distinct from himself each year
# we can access stuff by df[index[something]]

#remove everything before 1974
batters2 = batters2[batters2.yearID >= 1985]

# we need to combine same IDs, different stints into one person
# grouped = batters2.groupby(batters2.index)[['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'score']].sum()
bats2_grouped = batters2.groupby(batters2.index).agg({'playerID': np.max, 'yearID': np.max, 'G': np.sum, 'AB': np.sum, 'R': np.sum, 'H': np.sum, '2B': np.sum, '3B': np.sum, 'HR': np.sum, 'RBI': np.sum, 'SB': np.sum, 'CS': np.sum, 'BB': np.sum, 'SO': np.sum, 'score': np.sum, 'birthYear': np.max, 'nameFirst': np.max, 'nameLast': np.max, 'weight': np.max, 'height': np.max})

appearances = pd.read_csv('./data/Appearances.csv')

appearances = appearances[['yearID', 'playerID', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf', 'G_cf', 'G_rf', 'G_dh']]
# start at 1974 becuase DH position starts there
appearances = appearances[appearances.yearID >= 1985]

# add player/year ID to appearances
appearances_index = appearances.apply(create_player_year_id, axis=1)
appearances.set_index(appearances_index, inplace=True)
# remove player and year id from appearances so we can get max of positions
appearances.drop(['playerID', 'yearID'], axis=1, inplace=True)

# remove duplicates
appearances = appearances.groupby(appearances.index).sum()

def getPlayerPosition(row):
    if (appearances.loc[row.name].max() == 0):
        # probably a pitcher so remove
        return 'NA'
    else:
        return appearances.loc[row.name].idxmax()

# batters2 = batters2[batters2.index.isin(appearances.index)]
# batters2

bats2_grouped = bats2_grouped[bats2_grouped.index.isin(appearances.index)]

positions = bats2_grouped.apply(getPlayerPosition, axis=1)

# remove pitchers
bats2_grouped['position'] = positions

bats2_grouped = bats2_grouped[bats2_grouped['position'] != 'NA']

# add salaries
salaries = pd.read_csv('./data/Salaries.csv')

salary_index = salaries.apply(create_player_year_id, axis=1)
salaries.set_index(salary_index, inplace=True)

salaries.drop(['yearID', 'teamID', 'lgID', 'playerID'], axis=1, inplace=True)

bats2_final = pd.merge(left=bats2_grouped, right= salaries, how='left', left_index=True, right_index=True)

bats2_final.dropna(axis=0, how='any', inplace=True)

pitchers = pd.read_csv('./data/Pitching.csv')

pitchers = pitchers[pitchers['yearID'] >= 1985]

pitchers = pitchers[['playerID', 'yearID', 'teamID', 'W', 'L', 'G', 'SHO', 'SV', 'IPouts', 'H', 'ER', 'HR', 'BB', 'SO', 'BAOpp', 'R']]
pitchers_index = pitchers.apply(create_player_year_id, axis=1)
pitchers.set_index(pitchers_index, inplace=True)

# merge repeated entries from stints
pitchers_grouped = pitchers.groupby(pitchers.index).agg({'playerID': np.max, 'yearID': np.max, 'teamID': np.max, 'W': np.sum, 'L': np.sum, 'G': np.sum, 'SHO': np.sum, 'SV': np.sum, 'IPouts': np.sum, 'H': np.sum, 'ER': np.sum, 'HR': np.sum, 'BB': np.sum, 'SO': np.sum, 'BAOpp': np.mean, 'R': np.sum})

def pitcher_ERA(row):
    if row['IPouts'] == 0:
        return 15
    return (row['ER'] / row['IPouts']) * 27
    
def pitcher_score(row):
    return row['IPouts'] - (3 * row['ER']) - row['H'] - row['BB'] + row['SO'] + (5 * row['W'])

pitchers_grouped['ERA'] = pitchers_grouped.apply(lambda row: pitcher_ERA(row), axis=1)

pitchers_grouped['score'] = pitchers_grouped.apply(lambda row: pitcher_score(row), axis=1)

pitchers_grouped.sort_values(by = 'score', ascending=False, inplace=True)

pitchers2 = pd.merge(left=pitchers_grouped, right=master_df, how='left', left_on='playerID', right_on='playerID')

pitchers2_index = pitchers2.apply(create_player_year_id, axis=1)
pitchers2.set_index(pitchers2_index, inplace=True)

pitchers_complete = pd.merge(left=pitchers2, right= salaries, how='left', left_index=True, right_index=True)

pitchers_complete.dropna(axis=0, how='any', inplace=True)

pitchers_2010 = pitchers_complete[pitchers_complete["yearID"] == 2010]
pitchers_2010 = pitchers_2010.drop('score', axis=1)

pitchers_2011 = pitchers_complete[pitchers_complete["yearID"] == 2011]
pitchers_2011_rel = pitchers_2011[["playerID", "score"]]
pitchers_test = pd.merge(left=pitchers_2010, right=pitchers_2011_rel, left_on='playerID', right_on='playerID')

batters_2010 = bats2_final[bats2_final["yearID"] == 2010]
batters_2010 = batters_2010.drop('score', axis=1)

batters_2011 = bats2_final[bats2_final["yearID"] == 2011]
batters_2011 = batters_2011[["playerID", "score"]]

batters_test = pd.merge(left=batters_2010, right=batters_2011, left_on='playerID', right_on='playerID')

# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold
import numpy as np

# The columns we'll use to predict the target
pitcher_predictors = ["BB", "G", "H", "IPouts", "L", "HR", "BAOpp", "SO", 'W', "SV", "R", "ER", "SHO", "ERA", "weight", "height", "salary"]

# Initialize our algorithm class
alg = LinearRegression()

kf = KFold(pitchers_test.shape[0], n_folds=3, random_state=1)
pitcher_predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (pitchers_test[pitcher_predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = pitchers_test["score"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(pitchers_test[pitcher_predictors].iloc[test,:])
    pitcher_predictions.append(test_predictions)
                       
pitcher_predictions = np.concatenate(pitcher_predictions, axis=0)

prediction_diff = [abs(pitcher_predictions[i] - pitchers_test["score"][i]) for i in range(len(pitcher_predictions))]

batter_predictors = ["RBI", "H", "BB", "weight", "height", "HR", "R", "SO", "2B", "SB", "CS", "3B", "salary"]

kf2 = KFold(batters_test.shape[0], n_folds=3, random_state=1)
batter_predictions = []
for train, test in kf2:
    train_predictors = (batters_test[batter_predictors].iloc[train,:])
    train_target = batters_test["score"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(batters_test[batter_predictors].iloc[test,:])
    batter_predictions.append(test_predictions)
    
batter_predictions = np.concatenate(batter_predictions, axis=0)

prediction_diff = [abs(batter_predictions[i] - batters_test["score"][i]) for i in range(len(batter_predictions))]

pitchers_test["predicted_score"] = pitcher_predictions

batters_test["predicted_score"] = batter_predictions

def batter_to_tuple(row):
    return (row['playerID'], row['salary'], row['predicted_score'], row['position'])
def pitcher_team_to_tuple(row):
    return (row['teamID'], row['salary'], row['predicted_score'], 'pstaff')

batters_data = batters_test.apply(lambda row: batter_to_tuple(row), axis=1)

# pitching_teams = np.unique(pitchers_2011["teamID"])
pitchers_2011_teams = pitchers_2011[["playerID", "teamID"]]
pitchers_test = pitchers_test.drop('teamID', axis=1)

# update teams for next year
pitchers_test = pd.merge(left=pitchers_test, right=pitchers_2011_teams, left_on='playerID', right_on='playerID')

team_group = pitchers_test.groupby("teamID")


team_group = team_group.aggregate(np.sum)

# re-add index as column
team_group['teamID'] = team_group.index

pitchers_data = team_group.apply(lambda row: pitcher_team_to_tuple(row), axis=1)

masterlist = np.array(pitchers_data).tolist() + np.array(batters_data).tolist()

from constraint import *

problem = Problem()
positions = ['G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf', 'G_cf', 'G_rf', 'G_dh', 'pstaff']
position_domains = dict([])
pos_salaries_set = dict([])
pos_salaries = dict([])
for pos in positions:
    position_domains[pos] = []
    pos_salaries_set[pos] = set([])
    
for player in masterlist:
    position_domains[player[3]].append((player[0], player[2]))
    pos_salaries_set[player[3]].add(player[2])
    
for pos in positions:
    pos_salaries[pos] = list(pos_salaries_set[pos])

for pos in positions:
#     problem.addVariable(pos, position_domains[pos])
    problem.addVariable(pos + 'sal', pos_salaries[pos])

problem.addConstraint(MaxSumConstraint(500000000))
# print(problem.getSolutions())







