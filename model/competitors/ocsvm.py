from .base_model import BaseCompetitorModel
from sklearn.svm import OneClassSVM


class OCSVM(BaseCompetitorModel):
    def __init__(self, window_length, gpu=None):
        super().__init__(window_length, "OCSVM")

    def train(self, train_set):
        self.model = OneClassSVM(gamma='auto')
        self.model.fit(train_set)

    def test(self, test_set, window_label):
        pred = self.model.predict(test_set)
        pred[pred == 1] = 0
        pred[pred == -1] = 1
        scores = 1/(self.model.score_samples(test_set)+1e-8)

        window_pred = self.point_pred_to_window_pred(pred)
        window_scores = self.point_score_to_window_score(scores)
        real_window_label = window_label[:window_pred.shape[0]]

        return window_pred, window_scores, real_window_label
