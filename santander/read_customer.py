import pandas as pd
import re

def create_bag_of_spanish(gordo):
    # seem to be meaningful words in the column names
    bag_of_spanish = []
    for col in gordo.columns:
    
        # replace digits with nothing
        col = re.sub('\d','',col)
        # parse column name
        words = col.split('_')
    
        # add words to bag
        for w in words: 
            # filter out 'var' -- not meaningful
            not_var = re.search('^var', w) is None

            if w is not '' and not_var:
                bag_of_spanish.append(w)
    
    return list(set(bag_of_spanish))


def combine_spanish(gordo):

    print "Combining Spanish bag of words..."

    # get the set of computery spanish in the column names
    bag_of_spanish = create_bag_of_spanish(gordo)

    word_count = {}
    for i, one in gordo.iterrows():

        # for this customer, which positive (nonzero) features are present?
        features = list(one.iloc[one.nonzero()[0]].index)

        # look for any of the spanish bag words
        flags = {}
        for word in bag_of_spanish:
            
            col_with_word = [feat for feat in features if re.search(word, feat)]
            if len(col_with_word) > 0:

                # count number of matches with this word
                flags.update({ word: len(col_with_word) })

        # a row to the dictionary
        word_count.update({ i: flags })
    
    # bond together in a data frame
    spanish = pd.DataFrame(word_count).transpose()
    # fill in NaN
    spanish = spanish.fillna(0)

    print "Added %d Spanish columns" % spanish.shape[1]
    
    return spanish