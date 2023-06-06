import numpy as np
import random
import torch


class Transformation:
    def __init__(self, seed):
        self.seed = seed
        # random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def add_spike(X, scope: str = 'local', lbd=3, locality: int = 5, eps=1e-1):
        """
        Add a spike to a random index on a random dimension w.r.t. the local or global std

        :param X: input dataset window, shape (win_len, dim), format np.array
        :param scope: spike w.r.t. local context (local) or whole window (global)
        :param lbd: weighting parameter for locality
        :param locality: length of locality to be considered
        """
        assert scope in ['local', 'global'], f'Unknown parameter value scope={scope}.'

        spike_location = random.randint(locality, X.shape[0] - locality) if X.shape[0] > locality * 2 else X.shape[
                                                                                                               0] // 2
        spike_dim = random.randint(0, X.shape[1] - 1)
        locality_data = X[spike_location - locality:spike_location + locality, spike_dim] if scope == 'local' else X[:,
                                                                                                                   spike_dim]
        X[spike_location, spike_dim] = torch.mean(locality_data, axis=0) + (
                    torch.std(locality_data, axis=0) + eps) * lbd

        return X

    @staticmethod
    def shuffle(X):
        """
        Shuffle the first and second half of the time window

        :param X: input dataset window, shape (win_len, dim), format np.array
        :return: shuffled dataset window
        """

        mid = int(np.ceil(X.shape[0] / 2))
        if X.shape[0] % 2 != 0:
            idx = [i for i in range(mid, X.shape[0])] + [mid - 1] + [i for i in range(0, mid - 1)]
        else:
            idx = [i for i in range(mid, X.shape[0])] + [i for i in range(0, mid)]

        return torch.index_select(X, 0, torch.LongTensor(idx, device=X.device))

    @staticmethod
    def reset(X, eps=0.5):
        assert eps < 1, 'eps should smaller than 1'

        reset_start_location = random.randint(0, int(X.shape[0] * (1 - eps)))
        X[reset_start_location:reset_start_location + int(X.shape[0] * eps)] = 0

        return X

    @staticmethod
    def repeat(X):  # repeat the pattern with doubled frequency

        return torch.concat((X[::2], X[1:][::2]), axis=0)

    @staticmethod
    def trend(X):
        trend_factor = torch.tensor([[i / X.shape[0] for i in range(X.shape[0])] for _ in range(X.shape[1])],
                                    device=X.device).reshape(X.shape)
        return X * (trend_factor + 1)

    @staticmethod
    def add_noise(X, lbd: float = 0.01):
        """
        Add noise to X, depends on the overall min, max values and the scale factor lbd.
        :param X:
        :param lbd: scale in [0, 1]
        :return:
        """

        return X + (X.max() - X.min()) * torch.tensor(np.random.normal(loc=0., scale=lbd, size=X.shape),
                                                      device=X.device).float()

    @staticmethod
    def scale(X, lbd=0.1):
        scale_factor = torch.tensor(np.random.normal(loc=2.0, scale=lbd, size=X.shape), device=X.device)
        return torch.mul(X, scale_factor).float()
