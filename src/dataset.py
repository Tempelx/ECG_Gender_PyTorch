#!/usr/bin/env python
__author__ = "Felix Tempel"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import os

import numpy as np
import torch
import matplotlib
import scipy

from torch.utils.data import Dataset
import torchvision.transforms as tv

import src.augmentation as augmentation
import src.preprocessor as pp
from src.config import config_params


class GenderDataset(Dataset):
    def __init__(self, df_data, file_path, train):
        self.data = df_data.reset_index(drop=True)
        self.dimension = (int(config_params.dimension[0] * 100), int(config_params.dimension[1] * 100))

        self.train = train
        if self.train:
            self.augmenter = augmentation.Augmentation()

        self.file_path = file_path
        self.label_path = config_params.label_dir
        # no plotting
        matplotlib.use('Agg')

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, index):
        """
        Call the generator function to get one image.

        :param index: value in dataframe
        :return: X, y
        """
        path = os.path.join(self.file_path, str(self.data.recording[index]))
        signal_raw = scipy.io.loadmat(path)
        X, y = self.__generator_image(signal_raw, index)
        return X, y

    def __generator_image(self, signal_raw, index):
        """
        Generator is calling the function to generate the grayscale image.
        In case train == True data is augmented.

        :param index: value in dataframe
        :return: X, y
        """
        X = np.empty((1, *self.dimension), dtype=float)

        worker = pp.GenderLoader(train=self.train)
        signal = worker.execute_peak(signal_raw['val'], self.data.lead[index], self.data.r_peak_pos[index],
                                     self.data.r_peak_before[index], self.data.r_peak_after[index])
        # Augment
        if self.train:
            signal = self.augmenter.execute(signal, self.dimension, config_params.augment_prob, plot=False)
            signal = tv.transforms.ToTensor()(np.array(signal))
        else:
            signal = tv.transforms.ToTensor()(np.array(signal))
        X = signal

        if self.data.sex[index] == 1:
            # female
            label = [1]
        elif self.data.sex[index] == 0:
            # male
            label = [0]
        else:
            print("Fault in dataset! Label unknown")
        y = torch.tensor(label, dtype=float)

        return X, y

    def __generator_spectrogram(self, signal_raw, index):
        """
        Generator is calling the function to generate the spectrogram grayscale image.
        In case train == True data is augmented.

        :param index: value in dataframe
        :return: X, y
        """

        X = np.empty((1, *self.dimension), dtype=float)
        worker = pp.GenderLoader(train=self.train, win=config_params.window_size,
                                 hop=config_params.hop_size, nfft=config_params.nfft)

        # load data
        signal = worker.execute_spectro(signal_raw['val'], self.data.lead[index], self.data.r_peak_pos[index],
                                     self.data.r_peak_before[index], self.data.r_peak_after[index])
        # Augment
        if self.train:
            signal = self.augmenter.execute(signal, self.dimension, config_params.augment_prob)
            signal = tv.transforms.ToTensor()(np.array(signal))
        else:
            signal = tv.transforms.ToTensor()(np.array(signal))
        X = signal

        if self.data.sex[index] == 1:
            # female
            label = [1]
        elif self.data.sex[index] == 0:
            # male
            label = [0]
        else:
            print("Fault in dataset! Label unknown")
        y = torch.tensor(label, dtype=float)

        return X, y
