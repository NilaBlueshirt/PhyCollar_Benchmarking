import sys
import os
import random
import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import catboost as cat
from skbio import TreeNode
from collapse import PhyCollar

'''
a single line in the manifest file:
1st place - loop index: from 1 to xx, single integer
2nd place - model choice
output - one figure per outer loop
'''
# parsing inputs from the shell script
loop_index = sys.argv[1]
model_choice = sys.argv[2]
output_name = str(loop_index) + '_' + str(model_choice)
# read phylogenetic tree
tree = TreeNode.read('tree.nwk')

# read raw input files
data = pd.read_table('data.tsv', index_col=0)
meta = pd.read_table('meta.tsv', index_col=0)

def target(meta):
    name = 'Alzbiom: health vs Alzheimer\'s disease'
    prop = meta['AD'].astype(int)
    return prop, name
# use target.py
prop, name = target(meta)

# filter samples
ids = [x for x in data.columns if x in set(prop.index)]
data, prop = data[ids], prop.loc[ids]

# set up model input
X = data.values.T
y = prop.values
features = data.index.values

# CLR transform
X_ = np.log2(X + np.min(X[np.nonzero(X)]) / 2)
X_ -= X_.mean(axis=-1, keepdims=True)
####################################################################################################
# set up training process - using default parameters throughout the experiments
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
n = int(loop_index)
set_seed(n)
score_df = []  # text output of the AUC scores
cv = StratifiedKFold(n_splits=5, random_state=n, shuffle=True)
for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_train = X_[train_index]
    y_train = y[train_index]
    X_test = X_[test_index]
    y_test = y[test_index]
    X_train, y_train = ADASYN(random_state=n).fit_resample(X_train, y_train)
    if model_choice == 'cat':
        model = cat.CatBoostClassifier(random_seed=n)
    elif model_choice == 'mlp':  # sensitive to feature scaling
        model = MLPClassifier(random_state=n)
    elif model_choice == 'rf':
        model = RandomForestClassifier(random_state=n)
    elif model_choice == 'svm':
        model = SVC(random_state=n, kernel='rbf')
    else:
        pass
    tc = PhyCollar(estimator=model, tree=tree)
    tc.fit(X_train, y_train, feature_names=features)
    y_pred = tc.predict_proba(X_test, feature_names=features)
    # save the AUC score of each fold
    try:
        auc_fold = roc_auc_score(y_test, y_pred[:, 1])
    except:
        auc_fold = np.nan
    score_df.append({'Trial': loop_index, 'Fold': fold,  'AUC': auc_fold})
score_df = pd.DataFrame(score_df)
score_df.to_csv(str(output_name) + '_auc.csv', index=False)