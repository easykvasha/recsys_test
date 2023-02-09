from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import random

class DataPrep(ABC):
    """
    Interface class for DataPrep
    """

    @abstractmethod
    def data_preproc(self, data):
        return

class DataPrep_Users(DataPrep):

    """
    Data preprocessing for users.
    Get all information about users in csr_matrix

    """

    def __init__(self, user_region_data, user_age_data):
        self.user_region_data = user_region_data
        self.user_age_data = user_age_data

    @staticmethod
    def data_preparation(user_region_data):

        """
        Making user dictionary from user region or age
        param:
          user_region_data
        ---------
        return: dictionary with region or age information.
        """

        df = user_region_data.drop_duplicates(subset='row', keep="last")
        user_region_dict = pd.Series(df.col.values, index=df.row).to_dict()
        return user_region_dict

    def all_user_information(self, train_pivot_table, user_region_data, user_age_data):

        """
        Adding information about user region and age into users pivot table.
        param:
          train_pivot_table - users pivot table
          user_region_data - user region dataframe
          user_age_data - user age dataframe
        ---------
        return: numpy matrix in uint8 format
        """

        data_size = len(train_pivot_table.index)

        regions = np.zeros(data_size, dtype=np.uint8)
        ages = np.zeros(data_size, dtype=np.uint8)

        user_region_dict = self.data_preparation(user_region_data)
        user_age_dict = self.data_preparation(user_age_data)

        regions_list = user_region_data['col'].unique()
        ages_list = user_age_data['col'].unique()

        for idx, i in enumerate(train_pivot_table.index):

            if i in user_region_dict.keys():
                regions[idx] = user_region_dict[i] / 10
            else:
                regions[idx] = random.choice(regions_list)

            if i in user_age_dict.keys():
                ages[idx] = user_age_dict[i]
            else:
                ages[idx] = random.choice(ages_list)

        numpy_pivot_table = train_pivot_table.values.astype(np.uint8)

        for each in [regions, ages]:
            numpy_pivot_table = np.hstack(
                (numpy_pivot_table, each.reshape((data_size, 1))))

        return numpy_pivot_table

    def data_preproc(self, data):

        """Convert data from pandas df to sparse matrix"""

        train_pivot_table = data.pivot(
            index='row',
            columns='col',
            values='data',
        ).fillna(0)

        numpy_pivot_table = self.all_user_information(train_pivot_table, self.user_region_data, self.user_age_data)
        csr_matrix_users = csr_matrix(numpy_pivot_table)
        return csr_matrix_users, train_pivot_table.index


class DataPrep_Items(DataPrep):

    """
    Data preprocessing for items.
    Get all information about items in csr_matrix

    """

    def __init__(self, item_asset_data, item_price_data, item_subclass_data):
        self.item_asset_data = item_asset_data
        self.item_price_data = item_price_data
        self.item_subclass_data = item_subclass_data

    @staticmethod
    def data_preparation(data):

        """
        Collects all items in one dict
        param:
          data
        -----------
        return: dictionary containing items
        """

        df = data.drop_duplicates(subset='row', keep="last")
        item_asset_dict = pd.Series(df.data.values, index=df.row).to_dict()
        return item_asset_dict

    def all_items_information(self, train_pivot_table, item_asset_data, item_price_data, item_subclass_data):

        """
        Adding asset, price and subclass information into item-users pivot table.
        param:
          train_pivot_table - pivot table items as rows with values [0, 1] and users as columns,
          item_asset_data - item_asset dataframe
          item_price_data - item_price dataframe
          item_subclass_data - item_subclass dataframe
        ---------
        return: numpy matrix
        """

        data_size = len(train_pivot_table.index)

        assets = np.zeros(data_size, dtype=np.uint8)
        prices = np.zeros(data_size, dtype=np.uint8)
        subclasses = np.zeros(data_size, dtype=np.uint8)

        item_assets_dict = self.data_preparation(item_asset_data)
        item_prices_dict = self.data_preparation(item_price_data)
        item_subclasses_dict = self.data_preparation(item_subclass_data)

        assets_mean = item_asset_data['data'].mean()
        prices_mean = item_price_data['data'].mean()
        subclasses_mean = item_subclass_data['data'].mean()

        for idx, i in enumerate(train_pivot_table.index):

            try:
                assets[idx] = item_assets_dict[i]
            except IndexError:
                assets[idx] = assets_mean

            try:
                prices[idx] = item_prices_dict[i]
            except IndexError:
                prices[idx] = prices_mean

            try:
                subclasses[idx] = item_subclasses_dict[i]
            except IndexError:
                subclasses[idx] = subclasses_mean

        numpy_pivot_table = train_pivot_table.values.astype(np.uint8)

        for each in [assets, prices, subclasses]:
            numpy_pivot_table = np.hstack(
                (numpy_pivot_table, each.reshape((data_size, 1))))

        return numpy_pivot_table

    def data_preproc(self, data):

        """
        Convert data from pandas df to sparse matrix
        """

        train_pivot_table = data.pivot(
            index='col',
            columns='row',
            values='data'
        ).fillna(0)
        numpy_pivot_table = train_pivot_table.values.astype(np.uint8)
        csr_matrix_for_items = csr_matrix(numpy_pivot_table)

        return csr_matrix_for_items, train_pivot_table.index