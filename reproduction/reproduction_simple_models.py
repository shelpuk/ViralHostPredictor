import lightgbm as lgb
import numpy as np
import scipy
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score

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

data_array = experiment_data.iloc[:, 6:].to_numpy()
label_array = experiment_data['Reservoir'].to_numpy()

train_set, test_set, train_labels, test_labels = train_test_split(data_array, label_array, test_size=0.20, random_state=314, stratify=label_array)

# train_labels = preprocessing.label_binarize(train_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# test_labels = preprocessing.label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

lr = LogisticRegressionCV()

parameters = {'Cs':[1, 5, 10, 20, 50],
              'cv':[5],
              'penalty':['l2']}

clf = GridSearchCV(lr, parameters, cv=(test_set, test_labels))

clf.fit(train_set, train_labels)

print(accuracy_score(clf.predict(train_set), train_labels))
print(accuracy_score(clf.predict(test_set), test_labels))

