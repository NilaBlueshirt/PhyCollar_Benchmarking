import sys
import os
import random
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, RocCurveDisplay, auc, roc_auc_score
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgbm
import catboost as cat
# from skbio import TreeNode
# from collapse import PhyCollar


'''
a single line in the manifest file:
1st place - loop index: from 1 to xx, single integer
2nd place - model choice: RF, XGB, AdaB, LGBM, CAT, SVM, MLP
3th place - Use PCA or not: pca, none
output - one figure per outer loop
'''
# parsing inputs from the shell script
loop_index = sys.argv[1]
model_choice = sys.argv[2]
pca_choice = sys.argv[3]
output_name = str(loop_index) + '_' + str(model_choice) + '_' + str(pca_choice)
# read phylogenetic tree
# tree = TreeNode.read('tree.nwk')

####################################################################################################
# read raw input files
data = pd.read_table('data.tsv', index_col=0)
meta = pd.read_table('meta.tsv', index_col=0)

# target.py as in agp/sex.py
def target(meta):
    name = 'ASD - Dan 2020: stage (TD vs Autism)'
    cats = {'TD': 0, 'Autism': 1}
    prop = meta['Stage'].map(cats).dropna().astype(int)
    return prop, name
# use target.py
prop, name = target(meta)

# filter samples
ids = [x for x in data.columns if x in set(prop.index)]
data, prop = data[ids], prop.loc[ids]

# set up model input
X = data.values.T
y = prop.values
features = data.index.values  # not used here

# CLR transform
X = np.log2(X + np.min(X[np.nonzero(X)]) / 2)
X -= X.mean(axis=-1, keepdims=True)
####################################################################################################
# set up nested loop
# beginning of outer loop, for each outer loop n, do once split of the k-fold cv

# set global random seed, the actions in an independent trial use the same seed
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
n = int(loop_index)
set_seed(n)

# PCA value has to be within min(n_sample, n_feature) of each fold, due to limited samples PCA would compress column too much
# shouldn't do PCA whitening because there are correlations among OTUs
if pca_choice == 'pca':
    pca = PCA(n_components=100)

# setup outer cv loop
outer_cv = StratifiedKFold(n_splits=3, random_state=n, shuffle=True)
####################################################################################################
# begin outer cv loop: access each fold of the above one-time split
for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
    # set up train set from each fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    X_train, y_train = ADASYN(random_state=n).fit_resample(X_train, y_train)  # don't have to define n_neighbors
    # set up parameter grids for inner loop
    rf_parameters = {
        'n_estimators': [100, 150, 200],
        'max_features': [0.01, 0.1, 1, 10],
        'max_depth': [3, 10, 15],
        'min_samples_split': [2, 5, 10],  # default 2
        'min_samples_leaf': [1, 5, 10],  # default 1
        'max_samples': [0.1, 0.5, 1],  # default None
        'criterion': ['gini', 'entropy', 'log_loss']  # default gini
    }
    xgb_parameters = {
        "min_child_weight": [1, 5, 10],  # default 1
        "colsample_bytree": [0.1, 0.3, 0.6],
        "gamma": [0, 0.1, 0.5],  # default 0
        "learning_rate": [0.001, 0.01, 0.1, 1],  # default 0.3
        "max_depth": [3, 5, 20, 50],  # default 3
        "n_estimators": [100, 150, 200],  # default 100
        "subsample": [0.1, 0.3, 0.6]  # default 1
    }
    adab_parameters = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.001, 0.01, 0.1]
    }
    lgbm_parameters = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'num_leaves': [7, 15, 21]
    }
    cat_parameters = {
        'iterations': [100, 150, 200],
        'learning_rate': [0.001, 0.01, 0.1, 1],
        'depth': [3, 10, 50]
    }
    svm_parameters = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [10, 1, 0.1, 0.01, 'scale', 'auto'],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    mlp_parameters = {
        'hidden_layer_sizes': [50, 100, 150, 200],  # default 100
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.001, 0.01, 0.1, 1],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [50, 100, 200]
    }
    inner_scores = []
    inner_best_models = []
    ####################################################################################################
    # beginning of inner loop, for each split from outer loop, for each train-test pair, run once inner cv
    inner_cv = StratifiedKFold(n_splits=2, random_state=n, shuffle=True)
    if model_choice == 'RF':
        grid = GridSearchCV(
            RandomForestClassifier(random_state=n),  # don't use class_weight='balanced' due to up-sampling and stratified cv
            rf_parameters,
            cv=inner_cv,
            n_jobs=-1,
            scoring='roc_auc',
            error_score=0  # default error_score=np.nan will cause later average result to np.nan
        )
    elif model_choice == 'XGB':
        grid = GridSearchCV(
            xgb.XGBClassifier(seed=n),  # don't have to define objective, nthread may disrupt n_jobs
            xgb_parameters,
            cv=inner_cv,
            n_jobs=-1,
            scoring='roc_auc',
            error_score=0
        )
    elif model_choice == 'AdaB':
        grid = GridSearchCV(
            AdaBoostClassifier(random_state=n),
            adab_parameters,
            cv=inner_cv,
            n_jobs=-1,
            scoring='roc_auc',
            error_score=0
        )
    elif model_choice == 'LGBM':
        # doesn't use categorical_column here
        grid = GridSearchCV(
            lgbm.LGBMClassifier(random_state=n, class_weight='balanced'),
            lgbm_parameters,
            cv=inner_cv,
            n_jobs=-1,
            scoring='roc_auc',
            error_score=0
        )
    elif model_choice == 'CAT':
        grid = GridSearchCV(
            cat.CatBoostClassifier(random_seed=n, verbose=False),
            cat_parameters,
            cv=inner_cv,
            n_jobs=-1,
            scoring='roc_auc',
            error_score=0
        )
    elif model_choice == 'SVM':
        grid = GridSearchCV(
            SVC(random_state=n),
            svm_parameters,
            cv=inner_cv,
            n_jobs=-1,
            scoring='roc_auc',
            error_score=0
        )
    elif model_choice == 'MLP':
        grid = GridSearchCV(
            MLPClassifier(random_state=n),
            mlp_parameters,
            cv=inner_cv,
            n_jobs=-1,
            scoring='roc_auc',
            error_score=0
        )
    else:
        pass
    if pca_choice == 'pca':
        grid.fit(pca.fit_transform(X_train), y_train)
        y_pred = grid.predict_proba(pca.fit_transform(X_test))
    else:
        grid.fit(X_train, y_train)
        y_pred = grid.predict_proba(X_test)
    with open(output_name + '_report.txt', 'a') as f:
        print('For outer cv fold ' + str(fold) + ' with random state ' + str(n), file=f)
        print('Best parameters found: ', grid.best_params_, file=f)
        print('Best score: ', grid.best_score_, file=f)
        print('_' * 10)
    inner_best_models.append(grid.best_estimator_)
    inner_scores.append(grid.best_score_)  # max (mean score across 5 folds) for each loop
# end of the inner loop
###################################################################################################
# in case the inner loop run multiple times, more than one best estimator will produced by one outer fold
max_inner_score = np.max(inner_scores)
max_inner_score_index = np.argmax(inner_scores)  # return the first max item, max (mean score across all inner folds) of the current outer_cv fold
max_inner_model = inner_best_models[max_inner_score_index]  # best model after inner_loop, of the current outer_cv fold
