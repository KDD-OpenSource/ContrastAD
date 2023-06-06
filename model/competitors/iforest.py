from .base_model import BaseCompetitorModel
from sklearn.ensemble import IsolationForest
import numpy as np


class iForest(BaseCompetitorModel):
    def __init__(self, window_length, gpu=None):
        super().__init__(window_length, "iForest")

    def train(self, train_set):
        self.model = IsolationForest(random_state=0)
        self.model.fit(train_set)

    def test(self, test_set, window_label):
        pred = self.model.predict(test_set)
        pred[pred == 1] = 0
        pred[pred == -1] = 1

        window_pred = self.point_pred_to_window_pred(pred)
        window_scores = np.array([None for _ in range(window_pred.shape[0])])
        real_window_label = window_label[:window_pred.shape[0]]

        return window_pred, window_scores, real_window_label
