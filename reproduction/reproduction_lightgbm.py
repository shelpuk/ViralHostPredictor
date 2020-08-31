import lightgbm as lgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv('../BabayanEtAl_VirusData.csv', low_memory=False)

experiment_data = data[data['Reservoir'].isin(['Artiodactyl',
                                               'Carnivore',
                                               'Fish',
                                               'Galloanserae',
                                               'Insect',
                                               'Neoaves',
                                               'Plant',
                                               'Primate',
                                               'Pterobat',
                                               'Rodent',
                                               'Vespbat'])]

experiment_data = experiment_data.replace({'Reservoir': ['Artiodactyl', 'Carnivore', 'Fish', 'Galloanserae', 'Insect', 'Neoaves', 'Plant', 'Primate', 'Pterobat', 'Rodent', 'Vespbat']},
                                          {'Reservoir': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

# Verifying label replacement
print(experiment_data.Reservoir.unique())
print(experiment_data.shape)
print(experiment_data[['Virus name', 'Reservoir']].head())

data_array = experiment_data.iloc[:, 6:].to_numpy()
label_array = experiment_data['Reservoir'].to_numpy()

train_set, test_set, train_labels, test_labels = train_test_split(data_array, label_array, test_size=0.20, random_state=314, stratify=label_array)

# train_labels = preprocessing.label_binarize(train_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# test_labels = preprocessing.label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#
# train_set_frame = experiment_data.sample(frac=0.8, random_state=0)
# test_set_frame = experiment_data.drop(train_set_frame.index)
#
# train_labels = train_set_frame['Reservoir'].to_numpy()
# test_labels = test_set_frame['Reservoir'].to_numpy()
#
# train_set = train_set_frame.iloc[:, 6:].to_numpy()
# test_set = test_set_frame.iloc[:, 6:].to_numpy()

# train_data = lgb.Dataset(train_set, label=preprocessing.label_binarize(train_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
train_data = lgb.Dataset(train_set, label=train_labels)
test_data = lgb.Dataset(test_set, label=test_labels)

params = {"objective": "multiclass",
          "device" : "gpu",
          "num_class": 11,
          "num_leaves": 60,
          "max_depth": -1,
          "learning_rate": 0.01,
          "bagging_fraction": 0.9,
          "feature_fraction": 0.9,
          "bagging_freq": 5,
          "bagging_seed": 2018,
          "verbosity": -1}

num_round = 1000
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

predicted = bst.predict(test_set)

print(np.sum(np.argmax(predicted, axis=1)==test_labels)/test_labels.shape)

# param_test = {'num_leaves': [8, 16, 24, 32, 40, 48, 56, 64],
#               'min_child_samples': [100, 200, 300, 400, 500],
#               'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
#               'subsample': [0.1, 0.15, 0.2, 0.25, 0.3],
#               'colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6],
#               'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
#               'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
#               'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#               'bagging_fraction': [0.7, 0.8, 0.9, 0.95, 0.99],
#               'feature_fraction': [0.7, 0.8, 0.9, 0.95, 0.99],
#               'bagging_freq': [3, 4, 5, 6, 7]}
#
# fit_params = {'eval_metric': 'multi_logloss',
#               'eval_set': [(test_set, test_labels)],
#               'verbose': 100}
#
# clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, objective='multiclass_ova', num_class=11, silent=True, metric='None', n_jobs=4, n_estimators=5000)
# gs = RandomizedSearchCV(estimator=clf, param_distributions=param_test,
#                         n_iter=500,
#                         scoring='roc_auc',
#                         cv=3,
#                         refit=True,
#                         random_state=314,
#                         verbose=True)
#
# gs.fit(train_set, train_labels, **fit_params)
# print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))