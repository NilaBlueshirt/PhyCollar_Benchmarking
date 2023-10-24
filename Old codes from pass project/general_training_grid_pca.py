import sys
import os
import random
import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, RocCurveDisplay, auc, roc_auc_score
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgbm
import catboost as cat
import scipy
from scipy.stats import uniform, randint
import warnings
from sklearn import exceptions
warnings.filterwarnings(action="ignore", category=exceptions.DataConversionWarning)
warnings.filterwarnings(action="ignore", category=exceptions.DataDimensionalityWarning)
warnings.filterwarnings(action="ignore", category=exceptions.FitFailedWarning)
warnings.filterwarnings(action="ignore", category=exceptions.UndefinedMetricWarning)
warnings.filterwarnings(action="ignore", category=exceptions.EfficiencyWarning)
warnings.filterwarnings(action="ignore", category=exceptions.ConvergenceWarning)
warnings.filterwarnings(action="ignore", category=exceptions.SkipTestWarning)  # not in sklearn doc but can still use it
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=exceptions.DataConversionWarning)
warnings.simplefilter(action="ignore", category=exceptions.DataDimensionalityWarning)
warnings.simplefilter(action="ignore", category=exceptions.FitFailedWarning)
warnings.simplefilter(action="ignore", category=exceptions.UndefinedMetricWarning)
warnings.simplefilter(action="ignore", category=exceptions.EfficiencyWarning)
warnings.simplefilter(action="ignore", category=exceptions.ConvergenceWarning)
warnings.simplefilter(action="ignore", category=exceptions.SkipTestWarning)  # not in sklearn doc but can still use it
warnings.simplefilter("ignore")
# mamba install -c anaconda pandas numpy scikit-learn -y
# mamba install -c conda-forge imbalanced-learn matplotlib xgboost -y
# conda install -c conda-forge lightgbm
# conda install -c conda-forge catboost


'''
a single line in the manifest file:
1st place - loop index: from 1 to 50, single integer
2nd place - model choice: RF, XGB, AdaB, LGBM, CAT, SVM
3rd place - data partition: all, NN_CP, CN_CP
4th place - sampling: none, smote, adasyn
5th place - input file choice: genus_abs/nor/rel, family_..., order_... (not using abs currently)
6th place - scaler choice: log_otu/all, maxabs_otu/all, none
output - one figure per outer loop
'''
# parsing inputs from the shell script
loop_index = sys.argv[1]
model_choice = sys.argv[2]
data_choice = sys.argv[3]
sampling_choice = sys.argv[4]
file_choice = sys.argv[5]
scaler = sys.argv[6]
output_fig_name = str(loop_index) + '_' + str(model_choice) + '_' + str(data_choice) + \
                  '_' + str(sampling_choice) + '_' + str(file_choice) + '_' + str(scaler)

####################################################################################################
# select file, not using abs value for bad performance
# level genus-lowest
if file_choice == 'genus_abs':
    df = pd.read_csv('genus_abs_reads_otu_drop.csv')
elif file_choice == 'genus_nor':
    df = pd.read_csv('genus_normalized_reads_all.csv')
elif file_choice == 'genus_rel':
    df = pd.read_csv('genus_relative_abundance_all.csv')

# level family-mid
elif file_choice == 'family_abs':
    df = pd.read_csv('family_abs_reads_otu_drop.csv')
elif file_choice == 'family_nor':
    df = pd.read_csv('family_normalized_reads_all.csv')
elif file_choice == 'family_rel':
    df = pd.read_csv('family_relative_abundance_all.csv')

# level order-highest
elif file_choice == 'order_abs':
    df = pd.read_csv('order_abs_reads_otu_drop.csv')
elif file_choice == 'order_nor':
    df = pd.read_csv('order_normalized_reads_all.csv')
elif file_choice == 'order_rel':
    df = pd.read_csv('order_relative_abundance_all.csv')

else:
    print('Input file name not found.')

####################################################################################################
# select data partitions
df.patientID = df.patientID.astype('category').cat.codes
metadata_list = ['sampleID', 'EverCovid', 'CovidStatus', 'CovidLabel', 'Timepoint']  # keep patientID for later use
if data_choice == 'all':
    df_otu = df.drop(columns=metadata_list)
    X = df_otu.to_numpy()
    y = df['CovidStatus'].to_numpy()  # y is 0 or 1
    groups = df.patientID.to_list()  # for outer_cv use

elif data_choice == 'CN_CP':
    # Case-Neg from T1&2, Case-Pos from T3, samples are from the same person
    df0 = df.loc[df['Timepoint'].isin([1, 2, 3])]  # all three time points T1,2,3
    df1 = df0.loc[df0['CovidLabel'] == 1]  # case-negatives (CN) only exist in T1&2
    df2 = df0.loc[df0['CovidLabel'] == 2]  # case-positives (CP) only exist in T3
    # reformat y values from [1, 2] to [0, 1], to work with XGB
    df1['CovidLabel'] = 0
    df2['CovidLabel'] = 1
    df_CN_CP = pd.concat([df1, df2], ignore_index=True)  # reset the index after concat
    df_CN_CP_otu = df_CN_CP.drop(columns=metadata_list)
    X = df_CN_CP_otu.to_numpy()
    y = df_CN_CP['CovidLabel'].to_numpy()
    groups = df_CN_CP.patientID.to_list()  # for outer_cv use

elif data_choice == 'NN_CP':
    # Control from T1&2, Case-Pos from T3
    df0 = df.loc[df['Timepoint'].isin([1, 2])]  # select rows in T1, T2
    df1 = df0.loc[df0['CovidLabel'] == 0]  # controls (NN) exist in T1&2, not using the ones in T3
    df0 = df.loc[df['Timepoint'].isin([1, 2, 3])]  # all three time points T1,2,3
    df2 = df0.loc[df0['CovidLabel'] == 2]  # case-positives (CP) only exist in T3
    # reformat y values from [0, 2] to [0, 1], to work with XGB
    df2['CovidLabel'] = 1
    df_NN_CP = pd.concat([df1, df2], ignore_index=True)  # reset the index after concat
    df_NN_CP_otu = df_NN_CP.drop(columns=metadata_list)
    X = df_NN_CP_otu.to_numpy()
    y = df_NN_CP['CovidLabel'].to_numpy()
    groups = df_NN_CP.patientID.to_list()  # for outer_cv use

else:
    print('Input data partition name wrong.')

# X still has patientID column after this step, since inner_cv also need its own groups variable
####################################################################################################
# select scaler
if scaler == 'log_otu':
    # RF - Gut bowel disease paper only did log on OTU data, accoding to https://machinelearningmastery.com/selectively-scale-numerical-input-variables-for-machine-learning/ variables can be scaled selectively
    # add overall minimal value/2 to all the zero cells of otu cells
    # X[:, :-3] is otu, excluding the age, HIVStatus, patientID columns
    addons = np.min(X[:, :-3][np.nonzero(X[:, :-3])])/2  # find the min non-zero value, based on RF - Gut bowel disease paper
    X[:, :-3] += addons  # only add values & log transform on OTU data
    X[:, :-3] = np.log2(X[:, :-3])  # based on "A Fair Comparison" paper, they used log2

# log transform on OTU + Age, not using the patientID in the last column, HIVstatus is categorical column - no need to scale
elif scaler == 'log_all':
    addons = np.min(X[:, :-3][np.nonzero(X[:, :-3])])/2  # not using Age, HIVstatus, patientID
    X[:, :-2] += addons
    X[:, :-2] = np.log2(X[:, :-2])  # not scaling HIVstatus, patientID

# only transform OTU data
elif scaler == 'maxabs_otu':
    X[:, :-3] = preprocessing.MaxAbsScaler().fit_transform(X[:, :-3])

# on OTU + Age data
elif scaler == 'maxabs_all':
    X[:, :-2] = preprocessing.MaxAbsScaler().fit_transform(X[:, :-2])  # not scaling HIVstatus, patientID

else:  # scaler == 'none', according to https://datascience.stackexchange.com/questions/60950/is-it-necessary-to-normalize-data-for-xgboost/60954#60954 tree based methods don't need normalization
    pass


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
groups = [int(x) for x in groups]  # groups must be integers
outer_cv = StratifiedGroupKFold(n_splits=2, random_state=n, shuffle=True)
# set up output figure components
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(12, 12))
####################################################################################################
# begin outer cv loop: access each fold of the above one-time split
for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y, groups)):
    # set up train set from each fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    if sampling_choice == 'adasyn':  # increase minority class samples from 10% to 50% in train set
        X_train, y_train = ADASYN(random_state=n, n_neighbors=2).fit_resample(X_train, y_train)   # the patientID column also need to increase
    elif sampling_choice == 'smote':  # also increase
        X_train, y_train = SMOTE(random_state=n, k_neighbors=2).fit_resample(X_train, y_train)
    else:  # none
        pass
    # set up parameter grids for inner loop
    rf_parameters = {
        'n_estimators': [3, 5, 10],
        'max_features': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3, 5],  # default 2
        'min_samples_leaf': [1, 3, 5],  # default 1
        'max_samples': [0.1, 0.3, 0.5],  # default None
        'criterion': ['gini', 'entropy', 'log_loss']  # default gini
    }
    xgb_parameters = {
        "min_child_weight": [1, 5, 10],  # default 1
        "colsample_bytree": [0.1, 0.3, 0.6],
        "gamma": [0, 0.1, 0.5],  # default 0
        "learning_rate": [0.001, 0.005, 0.01, 0.1],  # default 0.3
        "max_depth": [3, 5, 20, 50],  # default 3
        "n_estimators": [30, 50, 100],  # default 100
        "subsample": [0.1, 0.3, 0.6]  # default 1
    }
    adab_parameters = {
        "n_estimators": [5, 10, 15, 20, 25, 50],
        "learning_rate": [0.001, 0.01, 0.1]
    }
    lgbm_parameters = {
        'n_estimators': [5, 10, 15, 20, 25, 50],
        'learning_rate': [0.001, 0.01, 0.1],
        'num_leaves': [7, 15, 21]
    }
    cat_parameters = {
        'iterations': [3, 5, 10],
        'learning_rate': [0.001, 0.005, 0.01],
        'depth': [3, 5, 7]
    }
    svm_parameters = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': [10, 1, 0.1, 0.01, 'scale', 'auto'],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    mlp_parameters = {
        'hidden_layer_sizes': [10, 20, 50, 100],  # default 100
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [10, 50, 100, 200]
    }
    inner_scores = []
    inner_best_models = []
    # PCA value has to be within min(n_sample, n_feature) of each fold, due to limited samples PCA would compress column too much
    # shouldn't do PCA whitening because there are correlations among OTUs
    pca = PCA(n_components=5)
    # groups are changed along with each X_train
    ####################################################################################################
    # beginning of inner loop, for each split from outer loop, for each train-test pair, run once inner cv
    for m in range(1, 2):
        groups = X_train[:, -1].tolist()  # patientID is at the last column
        groups = [int(x) for x in groups]  # turn float to int
        inner_cv = StratifiedGroupKFold(n_splits=2, random_state=n, shuffle=True)  # use the new groups variable
        if model_choice == 'RF':
            rf_grid = GridSearchCV(
                RandomForestClassifier(random_state=n),  # don't use class_weight='balanced' due to up-sampling and stratified cv
                rf_parameters,
                cv=inner_cv,
                n_jobs=-1,
                scoring='roc_auc',
                error_score=0  # default error_score=np.nan will cause later average result to np.nan
            )
            rf_grid.fit(pca.fit_transform(X_train[:, :-1]), y_train)  # not include the patientID at the last column of X_train
            y_pred = rf_grid.predict(pca.fit_transform(X_test[:, :-1]))  # not include the patientID at the last column of X_test
            with open(output_fig_name + '_report.txt', 'a') as f:
                print('For outer cv fold ' + str(fold) + ' with random state ' + str(n), file=f)
                print('Best RF parameters found: ', rf_grid.best_params_, file=f)
                print('\n', file=f)
                print('Best score: ', rf_grid.best_score_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('-----')
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(rf_grid.best_estimator_)
            inner_scores.append(rf_grid.best_score_)  # max (mean score across 5 folds) for each loop
        elif model_choice == 'XGB':
            xgb_grid = GridSearchCV(
                xgb.XGBClassifier(seed=n),  # don't have to define objective, nthread may disrupt n_jobs
                xgb_parameters,
                cv=inner_cv,
                n_jobs=-1,
                scoring='roc_auc',
                error_score=0
            )
            xgb_grid.fit(pca.fit_transform(X_train[:, :-1]), y_train)
            y_pred = xgb_grid.predict(pca.fit_transform(X_test[:, :-1]))
            with open(output_fig_name + '_report.txt', 'a') as f:
                print('For outer cv fold ' + str(fold) + ' with random state ' + str(n), file=f)
                print('Best XGB parameters found: ', xgb_grid.best_params_, file=f)
                print('\n', file=f)
                print('Best score: ', xgb_grid.best_score_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('-----')
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(xgb_grid.best_estimator_)
            inner_scores.append(xgb_grid.best_score_)
        elif model_choice == 'AdaB':
            adab_grid = GridSearchCV(
                AdaBoostClassifier(random_state=n),
                adab_parameters,
                cv=inner_cv,
                n_jobs=-1,
                scoring='roc_auc',
                error_score=0
            )
            adab_grid.fit(pca.fit_transform(X_train[:, :-1]), y_train)
            y_pred = adab_grid.predict(pca.fit_transform(X_test[:, :-1]))
            with open(output_fig_name + '_report.txt', 'a') as f:
                print('For outer cv fold ' + str(fold) + ' with random state ' + str(n), file=f)
                print('Best AdaBoost parameters found: ', adab_grid.best_params_, file=f)
                print('\n', file=f)
                print('Best score: ', adab_grid.best_score_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('-----')
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(adab_grid.best_estimator_)
            inner_scores.append(adab_grid.best_score_)
        elif model_choice == 'LGBM':
            categorical_column = X_train[:, -2].tolist()  # the HIVstatus column
            categorical_column = [int(x) for x in categorical_column]
            lgbm_grid = GridSearchCV(
                lgbm.LGBMClassifier(random_state=n, class_weight='balanced'),
                lgbm_parameters,
                cv=inner_cv,
                n_jobs=-1,
                scoring='roc_auc',
                error_score=0
            )
            lgbm_grid.fit(pca.fit_transform(X_train[:, :-1]), y_train, categorical_feature=categorical_column)
            y_pred = lgbm_grid.predict(pca.fit_transform(X_test[:, :-1]))
            with open(output_fig_name + '_report.txt', 'a') as f:
                print('For outer cv fold ' + str(fold) + ' with random state ' + str(n), file=f)
                print('Best LGBM parameters found: ', lgbm_grid.best_params_, file=f)
                print('\n', file=f)
                print('Best score: ', lgbm_grid.best_score_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('-----')
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(lgbm_grid.best_estimator_)
            inner_scores.append(lgbm_grid.best_score_)
        elif model_choice == 'CAT':
            cat_grid = GridSearchCV(
                cat.CatBoostClassifier(random_seed=n, verbose=False),
                cat_parameters,
                cv=inner_cv,
                n_jobs=-1,
                scoring='roc_auc',
                error_score=0
            )
            cat_grid.fit(pca.fit_transform(X_train[:, :-1]), y_train)
            y_pred = cat_grid.predict(pca.fit_transform(X_test[:, :-1]))
            with open(output_fig_name + '_report.txt', 'a') as f:
                print('For outer cv fold ' + str(fold) + ' with random state ' + str(n), file=f)
                print('Best catBoost parameters found: ', cat_grid.best_params_, file=f)
                print('\n', file=f)
                print('Best score: ', cat_grid.best_score_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('-----')
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(cat_grid.best_estimator_)
            inner_scores.append(cat_grid.best_score_)
        elif model_choice == 'SVM':
            svm_grid = GridSearchCV(
                SVC(random_state=n),
                svm_parameters,
                cv=inner_cv,
                n_jobs=-1,
                scoring='roc_auc',
                error_score=0
            )
            svm_grid.fit(pca.fit_transform(X_train[:, :-1]), y_train)
            y_pred = svm_grid.predict(pca.fit_transform(X_test[:, :-1]))
            with open(output_fig_name + '_report.txt', 'a') as f:
                print('For outer cv fold ' + str(fold) + ' with random state ' + str(n), file=f)
                print('Best SVM parameters found: ', svm_grid.best_params_, file=f)
                print('\n', file=f)
                print('Best score: ', svm_grid.best_score_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('-----')
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(svm_grid.best_estimator_)
            inner_scores.append(svm_grid.best_score_)  # max (mean score across 5 folds) for each loop
        elif model_choice == 'MLP':
            mlp_grid = GridSearchCV(
                MLPClassifier(random_state=n),
                mlp_parameters,
                cv=inner_cv,
                n_jobs=-1,
                scoring='roc_auc',
                error_score=0
            )
            mlp_grid.fit(pca.fit_transform(X_train[:, :-1]), y_train)
            y_pred = mlp_grid.predict(pca.fit_transform(X_test[:, :-1]))
            with open(output_fig_name + '_report.txt', 'a') as f:
                print('For outer cv fold ' + str(fold) + ' with random state ' + str(n), file=f)
                print('Best MLP parameters found: ', mlp_grid.best_params_, file=f)
                print('\n', file=f)
                print('Best score: ', mlp_grid.best_score_, file=f)
                print('\n', file=f)
                print('Scores using the above parameters: ', classification_report(y_test, y_pred), file=f)
                print('-----')
                print('\n', file=f)
                print('\n', file=f)
            inner_best_models.append(mlp_grid.best_estimator_)
            inner_scores.append(mlp_grid.best_score_)  # max (mean score across 5 folds) for each loop
        else:
            pass
    # end of the inner loop
    ###################################################################################################
    max_inner_score_index = np.argmax(inner_scores)  # return the first max item
    max_inner_model = inner_best_models[max_inner_score_index]  # best model after inner_loop, of the current outer_cv fold
    max_inner_model.fit(pca.fit_transform(X_train[:, :-1]), y_train)  # not include the patientID at the last column of X_train
    viz = RocCurveDisplay.from_estimator(
        max_inner_model,
        pca.fit_transform(X_test[:, :-1]),  # not include the patientID at the last column of X_test
        y_test,
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
# end of the outer cv loop
############################################################################################################
ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.")
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label)")
ax.axis("square")
ax.legend(loc="lower right")
plt.savefig(str(output_fig_name) + '.png')
# TODO: tabular output format to summarize results