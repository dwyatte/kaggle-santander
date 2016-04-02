import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

from santander.preprocessing import ColumnDropper
from santander.preprocessing import ZERO_VARIANCE_COLUMNS, CORRELATED_COLUMNS

filename = 'submission_nn.csv'

pipeline = Pipeline([
    ('cd', ColumnDropper(drop=ZERO_VARIANCE_COLUMNS+CORRELATED_COLUMNS)),
    ('std', StandardScaler())
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
submission = pd.DataFrame({'ID': ID_test, 'TARGET': y_pred[:, -1]})
submission.to_csv(filename, index=False)
print 'Wrote %s' % filename