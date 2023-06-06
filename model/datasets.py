import abc
import csv
import logging
import os
import pickle
import pandas as pd
import numpy as np
import ast
from pathlib import Path

base_path = Path(__file__).parent.parent


# Adapted from: https://github.com/KDD-OpenSource/DeepADoTS/blob/master/src/datasets/dataset.py
class Dataset:

    def __init__(self, name: str, file_name: str):
        self.name = name
        self.processed_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           '../../dataset/processed/', file_name))

        self._data = None
        self.logger = logging.getLogger(__name__)

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def load(self):
        """Load dataset"""

    def data(self) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """Return dataset, load if necessary"""
        if self._data is None:
            self.load()
        return self._data

    def save(self):
        pickle.dump(self._data, open(self.processed_path, 'wb'))


class ECG5000(Dataset):
    """0 is the outlier class. The training set is free of outliers."""

    def __init__(self, win_len, subset=''):
        super().__init__(name="ECG5000", file_name='')  # We do not need to load data from a file
        self.win_len = win_len
        self.subset = subset

    def load(self):
        train = pd.read_csv('dataset/ecg_5000/train.csv', header=None)
        test = pd.read_csv('dataset/ecg_5000/test.csv', header=None)

        train = train[(train.iloc[:, -1] == 1) | (train.iloc[:, -1] == 2) | (train.iloc[:, -1] == 4)]
        test_anomalies = test[(test.iloc[:, -1] == 3) | (test.iloc[:, -1] == 5)].iloc[:, :-1].values.reshape(-1, 1)
        x_train = train.iloc[:, :-1].values.reshape(-1, 1)
        y_train_win = np.array([0 if x in [1, 2, 4] else 1 for x in train.iloc[:, -1]]).reshape(-1, 1)
        y_train = np.array([y for _ in range(train.shape[1] - 1) for y in y_train_win]).reshape(-1, 1)
        x_test = test.iloc[:, :-1].values.reshape(-1, 1)
        y_test_win = np.array([0 if x in [1, 2, 4] else 1 for x in test.iloc[:, -1]]).ravel()
        y_test = np.array([y for _ in range(test.shape[1] - 1) for y in y_test_win]).ravel()
        y_test_win_original = test.iloc[:, -1].values.ravel()

        self._data = tuple(data for data in [x_train, y_train, x_test, y_test, y_test_win])


class UCR(Dataset):

    def __init__(self, win_len, subset=''):
        super().__init__(name="UCR", file_name='')  # We do not need to load data from a file
        self.win_len = win_len
        self.subset = subset

    def load(self):
        x_train = pd.read_csv(f'dataset/ucr/ucr_{self.subset}_train.csv', header=None).values.reshape(-1, 1)
        x_test = pd.read_csv(f'dataset/ucr/ucr_{self.subset}_test.csv', header=None).values.reshape(-1, 1)
        y_test = pd.read_csv(f'dataset/ucr/ucr_{self.subset}_label.csv', header=None).values.ravel()

        y_test_win = []
        for i in range(y_test.shape[0] // self.win_len):
            if sum(y_test[i * self.win_len:(i + 1) * self.win_len]) != 0:
                y_test_win.append(1)
            else:
                y_test_win.append(0)
        y_test_win = np.array(y_test_win).ravel()

        self._data = tuple(data for data in [x_train, None, x_test, y_test, y_test_win])


# Adepted from: https://github.com/imperial-qore/TranAD
class SWaT(Dataset):

    def __init__(self, win_len, subset=''):
        super().__init__(name="SWaT", file_name='')
        self.win_len = win_len
        self.subset = subset

    def normalize2(self, a, min_a=None, max_a=None):
        if min_a is None: min_a, max_a = min(a), max(a)
        return (a - min_a) / (max_a - min_a), min_a, max_a

    def load(self):
        dataset_folder = 'dataset/SWaT'
        file = os.path.join(dataset_folder, 'serie2.json')
        df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        df_test = pd.read_json(file, lines=True)[['val']][7000:12000]
        train, min_a, max_a = self.normalize2(df_train.values)
        test, _, _ = self.normalize2(df_test.values, min_a, max_a)
        labels = pd.read_json(file, lines=True)[['noti']][7000:12000]
        labels[labels == True] = 1
        labels[labels == False] = 0

        y_test_win = []
        for i in range(labels.shape[0] // self.win_len):
            if sum(labels.noti[i * self.win_len:(i + 1) * self.win_len]) != 0:
                y_test_win.append(1)
            else:
                y_test_win.append(0)
        y_test_win = np.array(y_test_win).ravel()

        self._data = tuple(data for data in [train.reshape(-1, 1),
                                             None,
                                             test.reshape(-1, 1),
                                             labels.values.ravel(),
                                             y_test_win])


class SMAP(Dataset):

    def __init__(self, win_len, subset=''):
        super().__init__(name="SMAP", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        x_train, y_train, x_test, y_test, y_test_binary = self.load_nasa('dataset/nasa', 'SMAP', self.win_len)

        self._data = tuple(data for data in [x_train, y_train, x_test, y_test, y_test_binary])

    def load_nasa(self, dataset_folder, dataset, win_len):
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        # label_folder = os.path.join(dataset_folder, 'test_label')
        # os.makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[0] == self.subset]
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels)

        x_train = self.concatenate_and_save(dataset_folder, 'train', data_info)
        x_test = self.concatenate_and_save(dataset_folder, 'test', data_info)

        y_train = np.array([0 for _ in range(x_train.shape[0])])
        y_test = np.array([1 if x > 0 else 0 for x in pd.Series(labels)
                          .groupby(np.arange(labels.size) // win_len).sum()])
        y_test_binary = y_test

        return x_train, y_train, x_test, y_test, y_test_binary

    def concatenate_and_save(self, dataset_folder, category, data_info):
        data = []
        for row in data_info:
            filename = row[0]
            temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
            data.extend(temp)
        data = np.asarray(data)

        return data


class MSL(Dataset):

    def __init__(self, win_len, subset=''):
        super().__init__(name="MSL", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        x_train, y_train, x_test, y_test, y_test_binary = self.load_nasa('dataset/nasa', 'MSL', self.win_len)

        self._data = tuple(data for data in [x_train, y_train, x_test, y_test, y_test_binary])

    def load_nasa(self, dataset_folder, dataset, win_len):
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        # label_folder = os.path.join(dataset_folder, 'test_label')
        # os.makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[0] == self.subset]
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels)

        x_train = self.concatenate_and_save(dataset_folder, 'train', data_info)
        x_test = self.concatenate_and_save(dataset_folder, 'test', data_info)

        y_train = np.array([0 for _ in range(x_train.shape[0])])
        y_test = np.array([1 if x > 0 else 0 for x in pd.Series(labels)
                          .groupby(np.arange(labels.size) // win_len).sum()])
        y_test_binary = y_test

        return x_train, y_train, x_test, y_test, y_test_binary

    def concatenate_and_save(self, dataset_folder, category, data_info):
        data = []
        for row in data_info:
            filename = row[0]
            temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
            data.extend(temp)
        data = np.asarray(data)

        return data
