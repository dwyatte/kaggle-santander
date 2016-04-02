import pandas as pd

filename = 'submission_ensemble.csv'

# xgb = pd.read_csv('submission_gbm.csv')
gbm = pd.read_csv('submission_features_gbm_adam.csv')
nn = pd.read_csv('submission_nn.csv')
knn_bag = pd.read_csv('submission_knn_bag.csv')

ensemble = gbm.copy()
ensemble['TARGET'] = gbm['TARGET']*0.8 + nn['TARGET']*0.1 + knn_bag['TARGET']*0.1
ensemble.to_csv(filename, index=False)
print 'Wrote %s' % filename