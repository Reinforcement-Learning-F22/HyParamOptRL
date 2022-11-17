import os
import torch
import numpy as np
import urllib.request
from torch.utils.data import DataLoader
from data_preprocessing import preprocessing_german
from utils import convert2torch


class GermanDataset(torch.utils.data.Dataset):
    """ Create traning data iterator """

    def __init__(self, feature_X, label_y, sensetive_a):
        self.X = feature_X.float()
        self.y = label_y.float()
        self.A = sensetive_a.float()
        if type(self.A) == np.ndarray:
            self.A = torch.from_numpy(self.A).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx, :], self.y[idx], self.A[idx]


class Dataset:
    def __init__(self, args_):
        self.test_data_loader = None
        self.train_data_loader = None
        self.train_data = None
        self.test_data = None
        self.S_test = None
        self.S_train = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.args = args_

    def dataset_preprocess(self, get_data=False):
        self.download_data()
        if self.args.dataset == 'German':
            X_train, X_test, y_train, y_test, S_train, S_test = \
                preprocessing_german(self.args.data_dir)
        else:
            raise Exception("Only German is available")
        X_train, X_test, y_train, y_test, S_train, S_test = convert2torch(X_train, X_test, y_train, y_test, S_train,
                                                                          S_test)
        self.X_train, self.X_test, self.y_train, self.y_test, self.S_train, self.S_test = \
            X_train, X_test, y_train, y_test, S_train, S_test
        if get_data:
            return (self.X_train, self.S_train), self.y_train, (self.X_test, self.S_test), self.y_test

    def download_data(self):
        if not os.path.exists(self.args.data_dir):
            os.makedirs(self.args.data_dir)
        if 'German' in self.args.dataset:
            if not os.path.isfile(f'{self.args.data_dir}/german.data'):
                print('Downloading german.data ...')
                urllib.request.urlretrieve(
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                    f'{self.args.data_dir}/german.data')
                print('Downloaded')
            else:
                if self.args.only_download_data:
                    print('german.data already downloaded')

    def get_dataset(self, return_=True):
        if any([self.X_train is None, self.X_test is None, self.y_train is None,
                self.y_test is None, self.S_train is None, self.S_test is None]):
            self.dataset_preprocess()
        self.train_data = GermanDataset(self.X_train, self.y_train, self.S_train)
        self.test_data = GermanDataset(self.X_test, self.y_test, self.S_test)
        if return_:
            return self.train_data, self.test_data

    def n_features(self):
        if self.X_train is None:
            self.dataset_preprocess()
        return self.X_train.shape[1]

    def dataset_size(self):
        if self.X_train is None:
            self.dataset_preprocess()
        return self.X_train.shape[0]

    def n_classes(self):
        if self.y_train is None:
            self.dataset_preprocess()
        return len(np.unique(self.y_train))

    def n_groups(self):
        if self.S_train is None:
            self.dataset_preprocess()
        return len(np.unique(self.S_train))
