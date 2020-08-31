import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd
from xgboost import XGBClassifier
import xgboost

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

train_set_frame = experiment_data.sample(frac=0.8, random_state=0)
test_set_frame = experiment_data.drop(train_set_frame.index)

train_labels = train_set_frame['Reservoir'].to_numpy()
test_labels = test_set_frame['Reservoir'].to_numpy()

train_set = train_set_frame.iloc[:, 6:].to_numpy()
test_set = test_set_frame.iloc[:, 6:].to_numpy()

class XGBoostTuner:
    def __init__(self, seed=0):
        self.seed=seed
        pass

    def _modelfit(self, alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            xgtrain = xgboost.DMatrix(X_train, label=y_train)
            cvresult = xgboost.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                                  metrics='merror', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

            n_estimators = cvresult.shape[0]
        return n_estimators

    def _init_xgb(self, learning_rate=0.1,
                  n_estimators=250,
                  max_depth=5,
                  min_child_weight=1,
                  gamma=0,
                  subsample=0.8,
                  colsample_bytree=0.8,
                  reg_alpha=0):

        return XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                             colsample_bynode=1, colsample_bytree=colsample_bytree, gamma=gamma,
                             learning_rate=learning_rate, max_delta_step=0, max_depth=max_depth,
                             min_child_weight=min_child_weight, missing=None, n_estimators=n_estimators, n_jobs=8,
                             nthread=8, objective='multi:softmax', random_state=0, reg_alpha=reg_alpha,
                             reg_lambda=1, scale_pos_weight=1, seed=self.seed, silent=None,
                             subsample=subsample, verbosity=1, num_class=11, tree_method='gpu_hist', gpu_id=1)

    def _run_grid_search(self, param_test, xgb_classifier, X_train, y_train):
        gsearch = GridSearchCV(estimator=xgb_classifier,
                               param_grid=param_test, scoring='accuracy', n_jobs=8, iid=False, cv=5)
        gsearch.fit(X_train, y_train)

        best_estimator = gsearch.best_estimator_
        params = best_estimator.get_params()

        best_params = {}
        for key in param_test.keys():
            best_params[key] = params[key]

        return best_params



    def tune(self, X_train, y_train):
        print("Start tune")
        xgb_classifier = self._init_xgb()

        # return xgb_classifier

        print("Modelfit")
        print(xgb_classifier)

        n_estimators = self._modelfit(xgb_classifier, X_train, y_train)

        print("Fit max_depth and min_child_weight")
        xgb_classifier = self._init_xgb(n_estimators=n_estimators)

        param_test = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2)}

        best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)

        optimal_max_depth = best_params['max_depth']
        optimal_min_child_weight = best_params['min_child_weight']

        xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight)

        print(xgb_classifier)

        print("Fit max_depth and min_child_weight")
        param_test = {
            'max_depth': [optimal_max_depth - 1, optimal_max_depth, optimal_max_depth + 1],
            'min_child_weight': [optimal_min_child_weight - 1, optimal_min_child_weight, optimal_min_child_weight + 1]}

        best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)

        optimal_max_depth = best_params['max_depth']
        optimal_min_child_weight = best_params['min_child_weight']
        xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight)
        print(xgb_classifier)
        print("Fit max_depth and min_child_weight")
        if optimal_max_depth == 10:
            param_test = {
                'max_depth': range(10, 15)}

            best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)

            optimal_max_depth = best_params['max_depth']
            xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                            max_depth=optimal_max_depth,
                                            min_child_weight=optimal_min_child_weight)

        if optimal_min_child_weight == 6:
            param_test = {
                'min_child_weight': range(6, 13, 1)}

            best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)
            optimal_min_child_weight = best_params['min_child_weight']
            xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                            max_depth=optimal_max_depth,
                                            min_child_weight=optimal_min_child_weight)

        xgb_classifier = self._init_xgb(n_estimators=500,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight)
        print(xgb_classifier)
        print("Modelfit")
        n_estimators = self._modelfit(xgb_classifier, X_train, y_train)

        print("Fit gamma")
        xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight)
        param_test = {
            'gamma': [i / 10.0 for i in range(0, 5)]}

        best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)
        optimal_gamma = best_params['gamma']

        xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight,
                                        gamma=optimal_gamma)
        print(xgb_classifier)
        print("Fit gamma")
        if optimal_gamma != 0:
            param_test = {
                'gamma': [optimal_gamma - 0.05, optimal_gamma, optimal_gamma + 0.05]}
            best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)
            optimal_gamma = best_params['gamma']

        xgb_classifier = self._init_xgb(n_estimators=500,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight,
                                        gamma=optimal_gamma)
        print(xgb_classifier)
        print("Modelfit")
        n_estimators = self._modelfit(xgb_classifier, X_train, y_train)

        print("Fit subsample and colsample_bytree")
        param_test = {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]}

        best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)
        optimal_subsample = best_params['subsample']
        optimal_colsample_bytree = best_params['colsample_bytree']

        xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight,
                                        gamma=optimal_gamma,
                                        subsample=optimal_subsample,
                                        colsample_bytree=optimal_colsample_bytree)

        print(xgb_classifier)
        print("Fit subsample and colsample_bytree")
        param_test = {
            'subsample': [optimal_subsample - 0.05, optimal_subsample, optimal_subsample + 0.05],
            'colsample_bytree': [optimal_colsample_bytree - 0.05, optimal_colsample_bytree,
                                 optimal_colsample_bytree + 0.05]}

        best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)
        optimal_subsample = best_params['subsample']
        optimal_colsample_bytree = best_params['colsample_bytree']

        xgb_classifier = self._init_xgb(n_estimators=500,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight,
                                        gamma=optimal_gamma,
                                        subsample=optimal_subsample,
                                        colsample_bytree=optimal_colsample_bytree)
        print(xgb_classifier)
        print("Modelfit")
        n_estimators = self._modelfit(xgb_classifier, X_train, y_train)

        print("Fit reg_alpha")
        xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight,
                                        gamma=optimal_gamma,
                                        subsample=optimal_subsample,
                                        colsample_bytree=optimal_colsample_bytree)

        param_test = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
        best_params = self._run_grid_search(param_test, xgb_classifier, X_train, y_train)
        optimal_reg_alpha = best_params['reg_alpha']

        xgb_classifier = self._init_xgb(n_estimators=1000,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight,
                                        gamma=optimal_gamma,
                                        subsample=optimal_subsample,
                                        colsample_bytree=optimal_colsample_bytree,
                                        reg_alpha=optimal_reg_alpha,
                                        learning_rate=0.01)

        print(xgb_classifier)
        print("Modelfit")
        n_estimators = self._modelfit(xgb_classifier, X_train, y_train, early_stopping_rounds=300)

        xgb_classifier = self._init_xgb(n_estimators=n_estimators,
                                        max_depth=optimal_max_depth,
                                        min_child_weight=optimal_min_child_weight,
                                        gamma=optimal_gamma,
                                        subsample=optimal_subsample,
                                        colsample_bytree=optimal_colsample_bytree,
                                        reg_alpha=optimal_reg_alpha,
                                        learning_rate=0.01)

        return xgb_classifier


t = XGBoostTuner()
xgb = t.tune(train_set, train_labels)

xgb.fit(train_set, train_labels)

print(xgb.score(train_set, train_labels))
print(xgb.score(test_set, test_labels))