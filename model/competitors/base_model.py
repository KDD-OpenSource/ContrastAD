import numpy as np
class BaseCompetitorModel:

    def __init__(self, window_length, name):
        self.model = None
        self.window_length = window_length
        self.name = name

    def train(self, train_set):
        raise NotImplementedError

    def test(self, test_set, window_label):
        raise NotImplementedError

    def point_score_to_window_score(self, scores):
        win_score = []
        for i in range(scores.shape[0] // self.window_length):
            win_score.append(scores[i * self.window_length:(i + 1) * self.window_length].max())
        return np.array(win_score)

    def point_pred_to_window_pred(self, pred):
        win_pred = []
        for i in range(pred.shape[0] // self.window_length):
            if pred[i * self.window_length:(i + 1) * self.window_length].sum() == 0:
                win_pred.append(0)
            else:
                win_pred.append(1)
        return np.array(win_pred)
