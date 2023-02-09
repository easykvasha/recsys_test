from Data_processing import DataPrep_Users
from Data_processing import DataPrep_Items
from Models import U2U
from Models import I2I
from Models import UserItemApproach
from Metric_calc import MetricsCalc

class GridSearch:

    def __init__(
            self,
            metrics_list,
            n_neighbours_list,
            data,
            test,
            item_asset,
            item_price,
            item_subclass,
            user_age,
            user_region):

        self.metrics_list = metrics_list
        self.n_neighbours_list = n_neighbours_list
        self.data = data
        self.test = test
        self.item_asset = item_asset
        self.item_price = item_price
        self.item_subclass = item_subclass
        self.user_age = user_age
        self.user_region = user_region
        self.best_score = 0
        self.grid_result = []

    def grid_search(self):
        print("Starting GridSearch")
        for metric in self.metrics_list:
            for n_neighbours in self.n_neighbours_list:
                print('Params: ' + metric + ' ' + str(n_neighbours))

                user_2_user = U2U(
                    self.data,
                    metric,
                    n_neighbours,
                    DataPrep_Users(self.user_region, self.user_age))

                item_2_item = I2I(
                    self.data,
                    metric,
                    n_neighbours,
                    DataPrep_Items(self.item_asset, self.item_price, self.item_subclass))

                combined = UserItemApproach(
                    self.data,
                    metric,
                    n_neighbours,
                    self.item_asset,
                    self.item_price,
                    self.item_subclass,
                    self.user_region,
                    self.user_age)

                user_score = MetricsCalc(user_2_user.predictions_of_model(), self.test).map_10()
                print('U2U_score: ' + str(user_score))

                item_score = MetricsCalc(item_2_item.predictions_of_model(), self.test).map_10()
                print('I2I_score: ' + str(item_score))

                combined_score = MetricsCalc(combined.predictions_of_model(), self.test).map_10()
                print('combined_score: ' + str(combined_score))

                if user_score > self.best_score:
                    self.grid_result = [metric, n_neighbours, 'user_2_user']
                    self.best_score = user_score

                if item_score > self.best_score:
                    self.grid_result = [metric, n_neighbours, 'item_2_item']
                    self.best_score = item_score

                if combined_score > self.best_score:
                    self.grid_result = [metric, n_neighbours, 'combined']
                    self.best_score = combined_score

        print('Best map@10 score {0} with params {1}'.format(self.best_score, self.grid_result))

