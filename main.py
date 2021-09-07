#!/usr/bin/env python
__author__ = "Felix Tempel"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import os
import sys
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from src.trainer import Trainer
from src.config import config_params
from src.dataset import GenderDataset
from src.utils import plot_confusion_matrix


def entry_train(config, args):
    """
    Entry function for training.

    :param config: dict
    :param args: args
    :return: None
    """
    CUDA = torch.cuda.is_available()
    path_labels = os.path.join(config_params.label_dir, args.database)
    df_dataset = pd.read_csv(path_labels)

    file_path = os.path.join(args.files)

    if args.database == "df_mit_op.csv":
        # MIT database no split!
        test_set = GenderDataset(df_dataset, file_path, train=False)
        test_generator = DataLoader(test_set, batch_size=int(config["batch_size"]),
                                    shuffle=True, num_workers=config_params.num_workers, drop_last=True)
        test_women = df_dataset.loc[df_dataset.sex == 1, 'sex'].count()
        test_men = df_dataset.loc[df_dataset.sex == 0, 'sex'].count()
        training_generator = None
        validation_generator = None
    else:
        # split 90% training 10% test
        train_set, test_set = train_test_split(df_dataset, test_size=0.10, random_state=0)
        # split training 70% training 30% validation
        train_set, val_set = train_test_split(train_set, test_size=0.3)
        # check proportions
        train_women = train_set.loc[df_dataset.sex == 1, 'sex'].count()
        train_men = train_set.loc[df_dataset.sex == 0, 'sex'].count()
        val_women = val_set.loc[df_dataset.sex == 1, 'sex'].count()
        val_men = val_set.loc[df_dataset.sex == 0, 'sex'].count()
        test_women = test_set.loc[df_dataset.sex == 1, 'sex'].count()
        test_men = test_set.loc[df_dataset.sex == 0, 'sex'].count()

        # Dataloader
        training_set = GenderDataset(train_set, file_path, train=True)
        training_generator = DataLoader(training_set, batch_size=int(config["batch_size"]),
                                        shuffle=True, num_workers=config_params.num_workers, drop_last=True)
        validation_set = GenderDataset(val_set, file_path, train=False)
        validation_generator = DataLoader(validation_set, batch_size=int(config["batch_size"]),
                                          shuffle=True, num_workers=config_params.num_workers, drop_last=True)
        test_set = GenderDataset(test_set, file_path, train=False)
        test_generator = DataLoader(test_set, batch_size=int(config["batch_size"]),
                                    shuffle=True, num_workers=config_params.num_workers, drop_last=True)

    # ResNet
    model = torch.hub.load('pytorch/vision:v0.6.0', config_params.model_name, pretrained=False)
    num_ftrs = model.fc.in_features
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_ftrs, out_features=1, bias=True)
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    train = Trainer(model=model, args=args, crit=criterion, optim=optimizer, train_dl=training_generator,
                    val_test_dl=validation_generator, test_dl=test_generator,
                    early_stopping_patience=config_params.early_stopping, cuda=CUDA)

    if args.command == "train":
        print('Start model training:')
        print("Model: \t" + config_params.model_name)
        print("Optimizer: \t " + str(optimizer))
        print("Batch_size: \t " + str(config["batch_size"]))
        print("Augmentation probability: " + str(config_params.augment_prob))
        print("Data trained: \t" + str(args.database))
        print("Train_datasize: ", len(training_set), " Val_datasize: ", len(validation_set))
        print("Train_datasize: Male:%d Female:%d \t Val_datasize: Male:%d Female:%d" % (train_men, train_women, val_men,
                                                                                        val_women))
        best_f1, best_acc = train.train()

    elif args.command == "test":
        print('Start model validation')
        print("Test_datasize: Male:%d Female:%d" % (test_men, test_women))
        path = args.ckpt
        if path is None:
            print("Please specify a path to the model weight file or train the data first!")
            sys.exit(0)
        conf_matrix, f1_test, auc_test, acc_test = train.val(path)
        import matplotlib
        matplotlib.use('TkAgg')
        plot_confusion_matrix(conf_matrix, norm=True, target=["Male", "Female"], title=str(args.database))
        breakpoint()

    elif args.command == "hypertrain":
        print('Start model hyper search training:')
        print("Model: \t" + config_params.model_name)
        print("Optimizer: \t " + str(optimizer))
        print("Augmentation probability: " + str(config_params.augment_prob))
        print("Data trained: \t" + str(args.database))
        print("Train_datasize: ", len(training_set), " Val_datasize: ", len(validation_set))
        print("Train_datasize: Male:%d Female:%d \t Val_datasize: Male:%d Female:%d" % (train_men, train_women, val_men,
                                                                                        val_women))
        best_f1, best_acc = train.train()

    else:
        print("Command not valid!")
        sys.exit(0)


def hyperparameter_search(args):
    """
    Hyperparameter function to ready ray tune.

    :param args: args
    :return: None
    """
    hyper_config = {
        "momentum": tune.choice([0.0, 0.4, 0.8, 0.9, 0.95]),
        "lr": tune.loguniform(1e-3, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "database": args.database
    }
    max_epochs = 20

    scheduler = ASHAScheduler(
        metric="mean_accuracy",
        mode="max",
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    if torch.cuda.is_available():
        res = {"cpu": 10, "gpu": 1}
    else:
        res = {"cpu": 10, "gpu": 0}

    result = tune.run(
        partial(entry_train, args=args),
        config=hyper_config,
        resources_per_trial=res,
        scheduler=scheduler,
        num_samples=5,
        progress_reporter=reporter,
        local_dir=config_params.hyper_dir,
        name="all_hp_search")

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.hyper_config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == '__main__':
    sys.argv = ['main.py', 'test', 'df_chin_op.csv', "./database", "--ckpt",
    "./ckpt/resnet34_28_10_2020__05_33/best_w.pth"]
    import argparse
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.set_default_tensor_type('torch.FloatTensor')
    parser = argparse.ArgumentParser(description="Model for Gender classification\n "
                                                 "type commands in the following way:\n"
                                                 "train df_mix_op.csv /User/...")
    parser.add_argument("command", metavar="<command>", help="train, hypertrain or test")
    parser.add_argument("database", metavar="<database>", help="Dataframe file location on your system")
    parser.add_argument("files", metavar="<files>", help="File location on your system")
    parser.add_argument("--ckpt", type=str, help="Path to model weights")
    parser.add_argument("--ex", type=str, help="Additional save name")
    args = parser.parse_args()

    if args.command == "hypertrain":
        hyperparameter_search(args)
    elif args.command == "train" or args.command == "test":
        config_dummy = {
            "momentum": 0.8,
            "lr": 0.0067906,
            "batch_size": 16,
            "database": args.database
        }
        entry_train(config_dummy, args)
    else:
        print("Command not valid!\n Please try train, test or hypertrain")
        sys.exit(0)
