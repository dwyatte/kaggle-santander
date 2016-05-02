import pandas as pd

from statsmodels.distributions import ECDF

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.covariance import EmpiricalCovariance
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from santander.preprocessing import ColumnDropper
from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS

filename = 'submission_elasticnet.csv'
heuristic_correction = True
bag = False

pipeline = Pipeline([
    ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS)),
    ('std', StandardScaler()),
])

df_train = pd.read_csv('data/train.csv')
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)
df_test = pd.read_csv('data/test.csv')
df_id = df_test['ID']
df_test = df_test.drop(['ID'], axis=1)

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
# m2 = ec.mahalanobis(df_train)
# df_train = df_train[m2 < 40000]
# df_target = df_target[m2 < 40000]
df_train['mahalanobis'] = ec.mahalanobis(df_train)
df_test['mahalanobis'] = ec.mahalanobis(df_test)

# clip
df_test = df_test.clip(df_train.min(), df_train.max(), axis=1)

# standard pipeline
pipeline = pipeline.fit(df_train)
X_train = pipeline.transform(df_train)
y_train = df_target
X_test = pipeline.transform(df_test)
ID_test = df_id

# params from cv experiments
if bag:
    el = BaggingClassifier(
        SGDClassifier(random_state=0, loss='log', penalty='elasticnet', learning_rate='invscaling',
                      eta0=0.1, alpha=0.001, n_iter=100),
        max_samples=0.999, max_features=0.999, n_estimators=10, random_state=0
    )
else:
    el = SGDClassifier(random_state=0, loss='log', penalty='elasticnet', learning_rate='invscaling',
                       eta0=0.1, alpha=0.001, n_iter=100)
el = el.fit(X_train, y_train)
print 'Final AUC: %f' % roc_auc_score(y_train, el.predict_proba(X_train)[:, -1])

y_pred = el.predict_proba(X_test)[:, -1]
if heuristic_correction:
    y_pred[age < 23] = 0

submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred})
submission.to_csv(filename, index=False)
print 'Wrote %s' % filename
