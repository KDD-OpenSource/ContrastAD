from .base_model import BaseCompetitorModel
from sklearn.neighbors import LocalOutlierFactor


class Lof(BaseCompetitorModel):
    def __init__(self, window_length, gpu=None):
        super().__init__(window_length, "LOF")

    def train(self, train_set):
        self.model = LocalOutlierFactor(n_neighbors=5)
        self.model.fit(train_set)

    def test(self, test_set, window_label):
        pred = self.model.fit_predict(test_set)
        pred[pred == 1] = 0
        pred[pred == -1] = 1
        scores = self.model.negative_outlier_factor_*-1

        window_pred = self.point_pred_to_window_pred(pred)
        window_scores = self.point_score_to_window_score(scores)
        real_window_label = window_label[:window_scores.shape[0]]

        return window_pred, window_scores, real_window_label
