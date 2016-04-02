import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from santander.preprocessing import ColumnDropper
from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS

filename = 'submission_knn_bag.csv'

pipeline = Pipeline([
    ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS)),
    ('std', StandardScaler()),
    ('pca', PCA(n_components=0.6))  # param from cv experiments
])

df_train = pd.read_csv('data/train.csv')
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)

pipeline = pipeline.fit(df_train)
X_train = pipeline.transform(df_train)
y_train = df_target

df_test = pd.read_csv('data/test.csv')
df_id = df_test['ID']
df_test = df_test.drop(['ID'], axis=1)
X_test = pipeline.transform(df_test)
ID_test = df_id

# params from cv experiments
knn_bag = BaggingClassifier(KNeighborsClassifier(n_jobs=-1),
                            max_samples=0.01, max_features=0.9, n_estimators=250, random_state=0)
knn_bag = knn_bag.fit(X_train, y_train)
print 'Final AUC: %f' % roc_auc_score(y_train, knn_bag.predict_proba(X_train)[:, -1])

y_pred = knn_bag.predict_proba(X_test)
submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred[:, -1]})
submission.to_csv(filename, index=False)
print 'Wrote %s' % filename