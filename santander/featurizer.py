import numpy as np
import pandas as pd

from math import log10
from statsmodels.distributions.empirical_distribution import ECDF

import read_customer as rc

MODULO = 1e3

def scale01(feature):
    ecdf = ECDF(feature)
    qtile = ecdf(feature)
    return qtile

def find_fake_balance_remainder(balance, modulo = MODULO, above_thresh = 1e2):
    remainder = balance % modulo
    value_count = remainder.value_counts()
    fake_balance = value_count / np.median(value_count) > above_thresh
    dup = value_count[fake_balance]
    fake_remainder = dup.index.values
    print "Removed %d instances of %0.2f" % (dup, fake_remainder)
    return fake_remainder

def is_legit(balance, fake_remainder, modulo = MODULO):
    fake_balance = balance % modulo in fake_remainder
    return not fake_balance

def filter_fakes(customer, col):
    # var38 with $117310.979016 is very duplicated -- should be removed
    balance = customer[col]
    fake = find_fake_balance_remainder(balance)
    
    should_proceed = [is_legit(b, fake) for b in balance]

    return customer[should_proceed]

def characterize(col, feature, min_uniq_len = 5):

    ### INTEGER VARIABLES
    if feature.dtype == 'int64':

        # check for few unique values
        len_uniq = len(feature.unique())
        
        if len_uniq < min_uniq_len:
            return pd.get_dummies(feature)
        
        elif max(customer[col] % 3) == 0:
            # lots of columns have uniques divisible by 3

            if max(customer[col]) > 999:
                # These are large round payments perhaps on a loan
                # ['imp_aport_var33_hace3', 'imp_aport_var33_ult1', 'var21']
                return scale01(feature)
            else:
                # these might be loan lifetimes like 3, 6, 12, 18, 24
                # they all start with 'num_'
                return scale01(feature)

        elif col == 'var3':
            # unclear gaussian mixture ????
            # if v > 25: delinquent = True
            return scale01(feature)

        elif col == 'var15':
            # probably age
            return scale01(feature)

        elif col == 'num_var4':
            # [0, 1, 2, 3, 4, 5, 6, 7] ????
            return pd.get_dummies(feature)

        elif col == 'var36':
            # [0, 1, 2, 3, 99] ????
            return pd.get_dummies(feature)

        else:
            print "Unknown int column ----> %s" % col
            return scale01(feature)


    ### FLOAT VARIABLES
    elif feature.dtype == 'float64':

        # does it have any negative values??
        is_negative = pd.DataFrame([int(f < 0) for f in feature])

        # what propotion of transactions end in .00?

        if max(is_negative) == 0:
            # strictly positive
            if col == 'var38':
                # might be customer lifetime value
                # non-missing, likely important
                return scale01(feature)
            else:
                return scale01(feature)
        else:
            feature_scaled = pd.DataFrame(scale01(feature))
            wide_feature = pd.concat([is_negative, feature_scaled], axis = 1)

            return wide_feature

    else:
        print "Feature %s is neither int nor float" % col
        return feature

def get_features(customer):

    frames = []
    for col, feature in customer.iteritems():
        wide_feature = pd.DataFrame(characterize(col, feature))
        frames.append(wide_feature)
        
    wide_customer = pd.concat(frames, axis = 1)

    print "Added %d features" % (wide_customer.shape[1] - customer.shape[1])
    return wide_customer

print "Reading data..."
df_train = pd.read_csv('../data/train.csv')
df_target = df_train['TARGET']
customer = df_train.drop(['TARGET', 'ID'], axis=1)

# get rid of bad rows
customer = filter_fakes(customer, 'var38')

# # combine spanish from column names
# spanish = rc.combine_spanish(customer)
# customer = pd.concat([customer, spanish], axis = 1)

# reverse feature engineering
customer = get_features(customer)

descriptive = customer.describe()
print descriptive.transpose()
