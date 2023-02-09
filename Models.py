from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
import random
from collections import Counter
from Data_processing import DataPrep_Users
from Data_processing import DataPrep_Items


class RecommendationModel(ABC):
    """
    An abstract base class for recommendation model.
    """

    def __init__(self, data, metrics, n_neighbours, data_processor):
        self.data = data
        self.metrics = metrics
        self.n_neighbours = n_neighbours
        self.data_processor = data_processor
        self.users_idx = []
        self.model = NearestNeighbors(metric=self.metrics, algorithm='brute', n_neighbors=self.n_neighbours, n_jobs=-1)

    def model_fit(self):

        """
        Fitting NearestNeighbors model.
        """

        prepare_data, users_idx = self.data_processor.data_preproc(data=self.data)
        self.model.fit(prepare_data)
        self.users_idx = users_idx
        return prepare_data

    def nn_for_all(self):

        """
        Finds nearest neighbour for all items/users.
        --------
        return: nearest neighbours
        """
        prepare_data = self.model_fit()
        return self.model.kneighbors(prepare_data, n_neighbors=self.n_neighbours)

    @abstractmethod
    def predictions_of_model(self):

        """
        Making predictions
        -------
        return:
        """
        return

class U2U(RecommendationModel):

    """
    Prediction model for User to User approach.
    Checking distance between users with provided metric.
    Making predictions for each user with predictions_counter function
    """

    def predictions_of_model(self):

        """
        Finding neighbours and calculating distances for each user.
        Adding neighbours products in recommendations.
        """
        distances, neighbours = self.nn_for_all()
        prediction_dict = dict()
        for idx, user in enumerate(self.users_idx):
            predictions = []
            for each in neighbours[idx][1::]:
                predictions += list(self.data.loc[self.data['row'] == each]['col'].values)
                if len(predictions) >= 10:
                    break
            prediction_dict[user] = predictions
        return prediction_dict

class I2I(RecommendationModel):

    """
    Prediction model for Item to Item approach.
    Finds nearest items for each Item.
    Counts appearance of item for each users item recommendation with distance weight.
    Recommends 10 items with most weights for user.
    """

    def make_prediction_for_user(self, prediction_dict):
        users_list = self.data['row'].unique()
        return_dict = dict()
        for each_user in users_list:
            counter = Counter()
            for each_item in self.data.loc[self.data['row'] == each_user]['col'].values:
                if each_item in prediction_dict.keys():
                    for each_predicted_item, item_distance in zip(prediction_dict[each_item][0],
                                                                  prediction_dict[each_item][1]):
                        counter[each_predicted_item] += 1 * (1 - item_distance)
            return_dict[each_user] = [x[0] for x in counter.most_common(10)]
        return return_dict

    def predictions_of_model(self):
        distances, neighbours = self.nn_for_all()
        prediction_dict = dict()
        for idx, item in enumerate(self.users_idx):
            prediction_dict[item] = (neighbours[idx], distances[idx])
        return self.make_prediction_for_user(prediction_dict)


class UserItemApproach(RecommendationModel):
    """
    Calculate predictions for user_to_user approach
    Calculate predictions for item_to_item approach
    Finding intersections between predictions for each user
    If intersection is less than 10, randomly add items from previous predictions
    """

    def __init__(self, data, metrics, n_neighbours, item_asset, item_price, item_subclass, user_region, user_age):
        self.user_2_user = U2U(
            data,
            metrics,
            n_neighbours,
            DataPrep_Users(user_region, user_age))

        self.item_2_item = I2I(
            data,
            metrics,
            n_neighbours,
            DataPrep_Items(item_asset, item_price, item_subclass))

    def get_intersections(self):
        combined_prediction = dict()
        users_answer = self.user_2_user.predictions_of_model()
        items_answer = self.item_2_item.predictions_of_model()

        for k, v in items_answer.items():
            updated_set = set.intersection(set(v), set(users_answer[k]))
            if len(updated_set) < 10:
                while not len(updated_set) == 10:
                    updated_set.add(random.choice(v + users_answer[k]))
            combined_prediction[k] = list(updated_set)
        return combined_prediction

    def predictions_of_model(self):
        return self.get_intersections()