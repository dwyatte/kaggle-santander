import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler


filename = 'submission_ensemble.csv'

# gbm = pd.read_csv('submission_gbm.csv')
# gbm = pd.read_csv('submission_features_gbm_adam.csv')
# gbm = pd.read_csv('submission_xgb_lalala.csv')
gbm = pd.read_csv('submission_xgb_lalala_under_23.csv')
# gbm = pd.read_csv('submission_xgb_lalala_to_the_top.csv')
# gbm = pd.read_csv('submission_xgb_lalala_to_the_top_v3_27.csv')
# nn = pd.read_csv('submission_nn_good_old.csv')
nn = pd.read_csv('submission_nn.csv')
knn = pd.read_csv('submission_knn.csv')
el = pd.read_csv('submission_elasticnet.csv')

ensemble = gbm.copy()

# ranking averaging doesn't seem to help
#
# ensemble['TARGET'] = rankdata(gbm['TARGET'])*0.8 + rankdata(nn['TARGET'])*0.1 + rankdata(knn['TARGET'])*0.1
# ensemble['TARGET'] = rankdata(gbm['TARGET']*0.8) + rankdata(nn['TARGET']*0.08) + rankdata(knn['TARGET']*0.08) + \
#                      rankdata(el['TARGET']*0.04)
# ensemble['TARGET'] = rankdata(gbm['TARGET']*0.8) + rankdata(nn['TARGET']*0.09) + rankdata(knn['TARGET']*0.09) + \
#                      rankdata(el['TARGET']*0.02)
# ensemble['TARGET'] = MinMaxScaler().fit_transform(ensemble['TARGET'])


# ensemble['TARGET'] = gbm['TARGET']*0.8 + nn['TARGET']*0.1 + knn['TARGET']*0.1
# ensemble['TARGET'] = gbm['TARGET']*0.8 + nn['TARGET']*0.08 + knn['TARGET']*0.08 + el['TARGET']*0.04
# ensemble['TARGET'] = gbm['TARGET']*0.8 + nn['TARGET']*0.09 + knn['TARGET']*0.09 + el['TARGET']*0.02
ensemble['TARGET'] = gbm['TARGET']*0.8 + nn['TARGET']*0.07 + knn['TARGET']*0.06 + el['TARGET']*0.07

ensemble.to_csv(filename, index=False)
print 'Wrote %s' % filename