import sys
import os
import importlib
import random
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import catboost as cat


'''
a single line in the manifest file:
1st place - loop index: from 1 to xx, single integer
2nd place - model choice
3nd place - use pca(0.95 explained) or not
4th place - the full file pathway to each target.py in each folder
'''
# parsing inputs from the shell script
trial_index = sys.argv[1]
model_choice = sys.argv[2]
pca_choice = sys.argv[3]
target_location = sys.argv[4]  # e.g. '/scratch/tianche5/PhyCollar/binary/agp/sex.py'

# read input data files
parent_folder = os.path.dirname(target_location)  # e.g. '/scratch/tianche5/PhyCollar/binary/agp'
data_file = str(parent_folder + '/data.tsv')
meta_file = str(parent_folder + '/meta.tsv')
data = pd.read_table(data_file, index_col=0)
meta = pd.read_table(meta_file, index_col=0)

# load each target.py from each sub-folder
target_file = os.path.basename(target_location)  # e.g. 'sex.py'
target_name = os.path.splitext(target_file)[0]  # e.g. 'sex'
sys.path.insert(1, parent_folder)  # to use the target.py files in other directories; don't change [0] for system use
target = importlib.import_module(str(target_name))  # to use a module from strings
prop, name = target.target(meta)  # use the target function in each target.py

# filter samples
ids = [x for x in data.columns if x in set(prop.index)]
data, prop = data[ids], prop.loc[ids]

# set up input for model
X = data.values.T
y = prop.values

# CLR transform using smallest value in X as the pesudo-count
X = np.log2(X + np.min(X[np.nonzero(X)]) / 2)
X -= X.mean(axis=-1, keepdims=True)

####################################################################################################
# set up training process - using default parameters throughout the experiments
# set the random state seed using random trial index
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
n = int(trial_index)
set_seed(n)

# start cross validation loop
auc_df = []  # text output of the AUC scores
pr_df = []  # text output of the PR AUC scores
cv = StratifiedKFold(n_splits=5, random_state=n, shuffle=True)
for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    try:
        X_train, y_train = ADASYN(random_state=n, sampling_strategy='minority').fit_resample(X_train, y_train)
    except:
        pass  # original labels are nearly balanced and no pesudo samples are made
    model = DummyClassifier()
    if model_choice == 'cat':
        model = cat.CatBoostClassifier(random_seed=n)
    elif model_choice == 'mlp':  # sensitive to feature scaling
        model = MLPClassifier(random_state=n)
    elif model_choice == 'rf':
        model = RandomForestClassifier(random_state=n, n_jobs=-1)
    elif model_choice == 'svm':
        model = SVC(random_state=n, kernel='rbf', probability=True)
    elif model_choice == 'lg':
        model = LogisticRegression(random_state=n, n_jobs=-1)
    else:
        pass
    if pca_choice == 'pca':
        pca = PCA(n_components=0.95, svd_solver='full')  # get 95% variance explained
        pca.fit(X_train)
        X_train =  pca.transform(X_train)
        X_test = pca.transform(X_test)
    else:
        pass
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    auc_fold = 0  # save the AUC score of each fold
    pr_fold = 0  # save the PR AUC score of each fold
    try:
        auc_fold = roc_auc_score(y_test, y_pred[:, 1])
    except:
        auc_fold = np.nan
    try:
        pr_fold = average_precision_score(y_test, y_pred[:, 1])  # it's better to use AP instead of auc() https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics
    except:
        pr_fold = np.nan
    auc_df.append({'Trial': trial_index, 'Fold': fold,  'AUC': auc_fold})
    pr_df.append({'Trial': trial_index, 'Fold': fold,  'PR-AUC': pr_fold})
auc_df = pd.DataFrame(auc_df)
pr_df = pd.DataFrame(pr_df)
output_path = str(parent_folder) + '/' + str(target_name) + '_' + str(model_choice) + '_' + str(pca_choice)
if not os.path.exists(output_path):
  os.mkdir(output_path)
else:
    pass
auc_df.to_csv(str(output_path) + '/' + str(trial_index) + '_auc.csv', index=False)
pr_df.to_csv(str(output_path) + '/' + str(trial_index) + '_pr.csv', index=False)
