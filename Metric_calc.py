import ml_metrics

class MetricsCalc:

    def __init__(self, prediction, test_dataframe):
        self.prediction = prediction
        self.test = test_dataframe

    def map_10(self):
        """
        Get map@10 metric value
        """
        test_results = []
        prediction_results = []
        for each in self.test['row'].unique():
            test_interactions = list(self.test.loc[self.test['row'] == each]['col'].values)
            if len(test_interactions) > 0 and each in self.prediction.keys():
                test_results.append(test_interactions)
                prediction_results.append(self.prediction[each])

        self.test = test_results
        self.prediction = prediction_results

        return ml_metrics.mapk(self.prediction, self.test, 10)