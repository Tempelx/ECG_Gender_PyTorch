#!/usr/bin/env python
__author__ = "Felix Tempel"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import os


class Config:

    root_code = r'.'
    # path to files
    root_data = r'.'
    hyper_dir = os.path.join(root_code, 'ckpt/ray_tune')
    label_dir = os.path.join(root_code, 'src/ref')

    num_workers = 10
    # for data
    dimension = (1.28, 1.28)
    window_size = 6
    hop_size = 3
    nfft = 12
    zero_burst = 10
    augment_prob = 0.2

    # for train
    model_name = 'resnet34'
    stage_epoch = [15, 25, 35]
    batch_size = 20
    num_classes = 2
    max_epoch = 35
    early_stopping = 5
    ckpt = 'ckpt'

    # optimizer
    lr = 0.01
    momentum = 0.95

    current_w = 'current_w.pth'
    best_w = 'best_w.pth'
    lr_decay = 10


config_params = Config()
