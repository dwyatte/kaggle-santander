import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from santander.preprocessing import ColumnDropper
from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS

filename = 'submission_gbm.csv'

pipeline = Pipeline([
    ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS)),
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

# hand-tuned params
# learning_rate = 0.025
# n_estimators = 400
# min_child_weight = 5
# max_depth = 4

# better params using columns column subsampling, longer training
# learning_rate = 0.02
# n_estimators = 500
# max_depth = 6
# subsample = 1
# colsample_bytree = 0.85
# min_child_weight = 1  # default

# best params so far using column/row subsampling, even longer training
learning_rate = 0.01
n_estimators = 800
max_depth = 6
subsample = 0.9
colsample_bytree = 0.85
min_child_weight = 1  # default

xgb = XGBClassifier(seed=0, learning_rate=learning_rate, n_estimators=n_estimators,
                    min_child_weight=min_child_weight, max_depth=max_depth,
                    colsample_bytree=colsample_bytree, subsample=subsample)
xgb = xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)], eval_metric='auc')

y_pred = xgb.predict_proba(X_test)
submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred[:, -1]})
submission.to_csv(filename, index=False)
print 'Wrote %s' % filename
