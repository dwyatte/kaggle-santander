import pandas as pd
from xgboost import XGBClassifier

df_train = pd.read_csv('data/train.csv')
df_target = df_train['TARGET']
df_train = df_train.drop(['TARGET', 'ID'], axis=1)

df_test = pd.read_csv('data/test.csv')
df_id = df_test['ID']
df_test = df_test.drop(['ID'], axis=1)

# hand-tuned params for now
learning_rate = 0.025
n_estimators = 400
min_child_weight = 5
max_depth = 4
xgb = XGBClassifier(seed=0, learning_rate=learning_rate, n_estimators=n_estimators,
                    min_child_weight=min_child_weight, max_depth=max_depth)
xgb = xgb.fit(df_train, df_target, eval_set=[(df_train, df_target)], eval_metric='auc')

y_pred = xgb.predict_proba(df_test)
submission = pd.DataFrame({'ID': df_id, 'TARGET': y_pred[:, 1]})
submission.to_csv('submission.csv', index=False)


