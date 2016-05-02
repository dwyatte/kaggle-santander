# single layer blend through logistic regression based on https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
# see also https://github.com/log0/vertebral/blob/master/stacked_generalization.py
# http://mlwave.com/kaggle-ensembling-guide/
#
# this is a failed attempt at stacking. this is basically correct as far as I can tell, but doesn't actually help in
# terms ot he the public leaderboard on the competition
#
# this takes about 30 min to run on my laptop


import pandas as pd
import numpy as np

from statsmodels.distributions import ECDF

from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from santander.preprocessing import ColumnDropper
from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS


########################################################################################################################
# data prep
########################################################################################################################
df_train = pd.read_csv('data/train.csv')
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)

df_test = pd.read_csv('data/test.csv')
df_id = df_test['ID']
df_test = df_test.drop(['ID'], axis=1)
ID_test = df_id.values

# save for heuristic correction
age = df_test['var15']
age_ecdf = ECDF(df_train['var15'])
df_train['var15'] = age_ecdf(df_train['var15'])
df_test['var15'] = age_ecdf(df_test['var15'])

# feature engineering
df_train.loc[df_train['var3'] == -999999.000000, 'var3'] = 2.0
df_train['num_zeros'] = (df_train == 0).sum(axis=1)
df_test.loc[df_train['var3'] == -999999.000000, 'var3'] = 2.0
df_test['num_zeros'] = (df_test == 0).sum(axis=1)

# outliers
ec = EmpiricalCovariance()
ec = ec.fit(df_train)
m2 = ec.mahalanobis(df_train)
df_train = df_train[m2 < 40000]
df_target = df_target[m2 < 40000]

# clip
# df_test = df_test.clip(df_train.min(), df_train.max(), axis=1)

# standard preprocessing
prep = Pipeline([
    ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS + CORRELATED_COLUMNS)),
    ('std', StandardScaler())
])

X_train = prep.fit_transform(df_train)
X_test = prep.transform(df_test)
y_train = df_target.values


########################################################################################################################
# gbm
########################################################################################################################
# gbm_learning_rate = 0.01
# gbm_n_estimators = 800
# gbm_max_depth = 6
# gbm_subsample = 0.9
# gbm_colsample_bytree = 0.85
# gbm_min_child_weight = 1  # default

gbm_learning_rate = 0.0202048
gbm_n_estimators = 560
gbm_max_depth = 5
gbm_subsample = 0.6815
gbm_colsample_bytree = 0.701
gbm_min_child_weight = 1  # default

gbm = XGBClassifier(seed=0, learning_rate=gbm_learning_rate, n_estimators=gbm_n_estimators,
                    min_child_weight=gbm_min_child_weight, max_depth=gbm_max_depth,
                    colsample_bytree=gbm_colsample_bytree, subsample=gbm_subsample)

########################################################################################################################
# nn
########################################################################################################################
nn = Sequential()
nn.add(Dense(32, input_shape=(X_train.shape[1],), activation='sigmoid'))
nn.add(Dropout(0.25))
nn.add(Dense(32, activation='sigmoid'))
nn.add(Dropout(0.25))
nn.add(Dense(1, activation='sigmoid'))

nn_sgd_lr = 0.1
nn_sgd_decay = 1e-6
nn_sgd_momentum = 0.9

opt = SGD(lr=nn_sgd_lr, decay=nn_sgd_decay, momentum=nn_sgd_momentum, nesterov=True)
nn.compile(loss='binary_crossentropy', optimizer=opt)


########################################################################################################################
# knn bag
########################################################################################################################
knn_bag_pca_n_components = 0.6
knn_bag_max_samples = 0.01
knn_bag_max_features = 0.9
knn_bag_n_estimators = 250

knn_bag = Pipeline([
    ('pca', PCA(n_components=knn_bag_pca_n_components)),
    ('knn_bag', BaggingClassifier(KNeighborsClassifier(n_jobs=-1),
                                  max_samples=knn_bag_max_samples, max_features=knn_bag_max_features,
                                  n_estimators=knn_bag_n_estimators, random_state=0))
])


########################################################################################################################
# elasticnet
########################################################################################################################
el_eta0 = 0.1
el_alpha = 0.001
el_n_iter = 100

el = SGDClassifier(random_state=0, loss='log', penalty='elasticnet', learning_rate='invscaling',
                   eta0=0.1, alpha=0.001, n_iter=100)


########################################################################################################################
# blending
########################################################################################################################
filename = 'submission_stack.csv'
heuristic_correction = True
n_folds = 5

clfs = [gbm, nn, knn_bag, el]
skf = list(StratifiedKFold(y_train, n_folds))

print 'Creating train and test sets for blending'
dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        X_train_fold = X_train[train, :]
        y_train_fold = y_train[train]
        X_test_fold = X_train[test, :]
        y_test_fold = y_train[test]

        # nn hard-coded to run for 100 epochs using params defined above
        if type(clf) is Sequential:
            np.random.seed(1234)
            clf.fit(X_train_fold, y_train_fold, verbose=0, nb_epoch=100)
        else:
            clf.fit(X_train_fold, y_train_fold)

        y_pred = clf.predict_proba(X_test_fold)[:, -1]
        dataset_blend_train[test, j] = y_pred
        dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:, -1]
        print 'Fold %d, AUC: %f' % (i, roc_auc_score(y_test_fold, y_pred))
    # mean over folds
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)


########################################################################################################################
# stacking
########################################################################################################################

# L1 stacking would be improved by actually doing another proper kfold

# gbm w/ same params
std = StandardScaler()
dataset_blend_train = std.fit_transform(dataset_blend_train)
dataset_blend_test = std.transform(dataset_blend_test)
X_train_l1 = np.hstack([X_train, dataset_blend_train])
X_test_l1 = np.hstack([X_test, dataset_blend_test])

print 'GBM L1'
gbm_l1 = XGBClassifier(seed=0, learning_rate=gbm_learning_rate, n_estimators=gbm_n_estimators,
                       min_child_weight=gbm_min_child_weight, max_depth=gbm_max_depth,
                       colsample_bytree=gbm_colsample_bytree, subsample=gbm_subsample)
gbm_l1.fit(X_train_l1, y_train)
print 'GBM L1 AUC: %f' % roc_auc_score(y_train, gbm_l1.predict_proba(X_train_l1)[:, -1])

# nn w/ same params
nn_l1 = Sequential()
nn_l1.add(Dense(32, input_shape=(X_train_l1.shape[1],), activation='sigmoid'))
nn_l1.add(Dropout(0.25))
nn_l1.add(Dense(32, activation='sigmoid'))
nn_l1.add(Dropout(0.25))
nn_l1.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=nn_sgd_lr, decay=nn_sgd_decay, momentum=nn_sgd_momentum, nesterov=True)
nn_l1.compile(loss='binary_crossentropy', optimizer=opt)

print 'NN L1'
nn_l1.fit(X_train_l1, y_train, verbose=0, nb_epoch=100)
print 'NN L1 AUC: %f' % roc_auc_score(y_train, nn_l1.predict_proba(X_train_l1)[:, -1])

# final lr-blending
l1_pred_train = np.vstack([gbm_l1.predict_proba(X_train_l1)[:, -1], nn_l1.predict_proba(X_train_l1)[:, -1]]).T
l1_pred_test = np.vstack([gbm_l1.predict_proba(X_test_l1)[:, -1], nn_l1.predict_proba(X_test_l1)[:, -1]]).T
lr = LogisticRegression()
lr = lr.fit(l1_pred_train, y_train)
print 'Final LR AUC: %f' % roc_auc_score(y_train, lr.predict_proba(l1_pred_train)[:, -1])
y_pred = lr.predict_proba(l1_pred_test)[:, -1]

# # final hand-blending
# print 'Final Blended AUC: %f' % roc_auc_score(y_train, 0.8*gbm_l1.predict_proba(X_train_l1)[:, -1] +
#                                                        0.2*nn_l1.predict_proba(X_train_l1)[:, -1])
# y_pred = 0.8*gbm_l1.predict_proba(X_test_l1)[:, -1] + 0.2*nn_l1.predict_proba(X_test_l1)[:, -1]

if heuristic_correction :
    y_pred[age < 23] = 0

stack = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred})
stack.to_csv(filename, index=False)

print 'Wrote %s' % filename
