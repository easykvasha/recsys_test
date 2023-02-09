import pandas as pd
from sklearn.model_selection import train_test_split
from GridSearch import GridSearch


"""
Reading data
"""
interactions = pd.read_csv('data/interactions.csv')
item_subclass = pd.read_csv('data/item_subclass.csv')
item_price = pd.read_csv('data/item_price.csv')
item_asset = pd.read_csv('data/item_asset.csv')
user_age = pd.read_csv('data/user_age.csv')
user_region = pd.read_csv('data/user_region.csv')


"""
Splitting data to train and test
"""

train, test = train_test_split(interactions, test_size=0.2, random_state=42, shuffle=True)

mismatch_train = set(item_subclass['row']).difference(set(train['col'].values))
new_df_train = pd.DataFrame(data={
    'row': [0 for _ in mismatch_train],
    'col': list(mismatch_train),
    'data': [0 for _ in mismatch_train]})
train = pd.concat([train, new_df_train])

mismatch_test = set(item_subclass['row']).difference(set(test['col'].values))
new_df_test = pd.DataFrame(data={
    'row': [0 for _ in mismatch_test],
    'col': list(mismatch_test),
    'data': [0 for _ in mismatch_test]})
test = pd.concat([test, new_df_test])

""" GridSearch params """

metrics = ['cosine', 'euclidean']
n_neighbours_list = [3, 4]


"""Grid search """
clf = GridSearch(metrics,
                 n_neighbours_list,
                 train,
                 test,
                 item_asset,
                 item_price,
                 item_subclass,
                 user_age,
                 user_region)

clf.grid_search()