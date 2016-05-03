import numpy as np
import pandas as pd

from statsmodels.distributions import ECDF

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from santander.preprocessing import ColumnDropper
from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS

filename = 'submission_nn.csv'
heuristic_correction = True

np.random.seed(1234)

pipeline = Pipeline([
    ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS)),
    ('std', StandardScaler())
])

df_train = pd.read_csv('data/train.csv')
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)
df_test = pd.read_csv('data/test.csv')
df_id = df_test['ID']
df_test = df_test.drop(['ID'], axis=1)

# save for heuristic correction
age = df_test['var15']
# age_ecdf = ECDF(df_train['var15'])
# df_train['var15'] = age_ecdf(df_train['var15'])
# df_test['var15'] = age_ecdf(df_test['var15'])

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

# clip -- might not actually help on LB? or maybe it was new script
# df_test = df_test.clip(df_train.min(), df_train.max(), axis=1)

# standard pipeline
pipeline = pipeline.fit(df_train)
X_train = pipeline.transform(df_train)
y_train = df_target
X_test = pipeline.transform(df_test)
ID_test = df_id

model = Sequential()
model.add(Dense(32, input_shape=(X_train.shape[1],), activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

nb_epoch = 100
opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=opt)
model.fit(X_train, y_train, nb_epoch=nb_epoch)
print 'Final AUC: %f' % roc_auc_score(y_train, model.predict_proba(X_train))

y_pred = model.predict_proba(X_test)
if heuristic_correction:
    y_pred[age < 23] = 0

submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred[:, -1]})
submission.to_csv(filename, index=False)
print 'Wrote %s' % filename
