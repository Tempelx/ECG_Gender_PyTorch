#!/usr/bin/env python
__author__ = "Felix Tempel"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import os
import shutil
import time

import numpy as np
import torch as t
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, auc, roc_curve
from torch.utils.tensorboard import SummaryWriter
from ray import tune

from src.config import config_params


class Trainer:
    def __init__(self,
                 model,             # Model to be trained.
                 crit,              # Loss function
                 args,              # args
                 optim=None,        # Optimizer
                 train_dl=None,     # Training data set
                 val_test_dl=None,  # Validation data set
                 test_dl=None,      # Test data set
                 cuda=True,         # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._args = args
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._test_dl = test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def _save(self, state, best, model_save_dir):
        """
        Save the model's current and best weights.

        :param state: current state
        :param best: boolean
        :param model_save_dir: path to directory
        :return: None
        """
        current_weights = os.path.join(model_save_dir, config_params.current_w)
        best_weights = os.path.join(model_save_dir, config_params.best_w)
        t.save(state, current_weights)
        if best:
            shutil.copyfile(current_weights, best_weights)

    def _time(self, since):
        """
        Calculate the elapsed time.

        :param since: start time
        :return: elapsed time
        """
        time_elapsed = time.time() - since
        return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)

    def _adjust_lr(self, optimizer, lr):
        """
        Adjust the learning rate in the optimizer.

        :param optimizer:
        :param lr: float
        :return: lr
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _detach_tensor(self, tensor):
        """
        Detach the tensor for calculation.

        :param tensor:
        :return: detached tensor
        """
        return tensor.view(-1).cpu().detach().numpy()

    def _confusion_matrix(self, y_true, y_pre):
        """
        Calculate the confusion matrix - tensor already detached.

        :param y_true: tensor with ground truth values
        :param y_pre: tensor with predicted values
        :return: confusion matrix
        """
        return confusion_matrix(y_true, y_pre)

    def _list_build(self, y_true, y_pre, threshold=0.5):
        """
        Detach tensors and return for validation.

        :param y_true: tensor with ground truth values
        :param y_pre: tensor with predicted values
        :param threshold:
        :return: detached tensor
        """
        y_true = self._detach_tensor(y_true).astype(np.int)
        y_pre = self._detach_tensor(y_pre) > threshold
        return y_true, y_pre

    def _calc_f1(self, y_true, y_pre, threshold=0.5):
        """
        Calculate the F1 Score.

        :param y_true: tensor with ground truth values
        :param y_pre: tensor with predicted values
        :param threshold: 0.5
        :return: F1 score
        """
        y_true = self._detach_tensor(y_true).astype(np.int)
        y_pre = self._detach_tensor(y_pre) > threshold
        return f1_score(y_true, y_pre)

    def _calc_acc(self, y_true, y_pre, threshold=0.5):
        """
        Calculate the accuracy.

        :param y_true: tensor with ground truth values
        :param y_pre: tensor with predicted values
        :param threshold: 0.5
        :return: accuracy score
        """
        y_true = self._detach_tensor(y_true).astype(np.int)
        y_pre = self._detach_tensor(y_pre) > threshold
        return accuracy_score(y_true, y_pre)

    def _calc_auc(self, y_true, y_pre, threshold=0.5):
        """
        Calculate the AUC curve.

        :param y_true: tensor with ground truth values
        :param y_pre: tensor with predicted values
        :param threshold: 0.5
        :return: AUC curve
        """
        y_true = self._detach_tensor(y_true).astype(np.int)
        y_pre = self._detach_tensor(y_pre) > threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_pre)
        return auc(fpr, tpr)

    def _train_epoch(self):
        """
        Train the model for an epoch by looping over the batches.

        :return: train_loss, train_f1, train_auc
        """
        f1_meter, auc_meter, acc_meter, loss_meter, it_count, correct = 0, 0, 0, 0, 0, 0
        # set training mode
        self._model.train()
        # iterate through the training set
        for images, labels in self._train_dl:
            if self._cuda:
                # transfer the batch to the gpu if given
                images = images.to(t.device("cuda"))
                labels = labels.to(t.device("cuda"))
            # perform a training step
            self._optim.zero_grad()
            # propagate through the network
            output = self._model(images)
            # calculate the loss
            loss = self._crit(output, labels)
            # compute gradient by backward propagation
            loss.backward()
            # update weights
            self._optim.step()
            loss_meter += loss.item()
            it_count += 1
            f1_calc = self._calc_f1(labels, t.sigmoid(output))
            f1_meter += f1_calc
            acc_meter = self._calc_acc(labels, t.sigmoid(output))
            acc_meter += acc_meter
            auc_calc = self._calc_auc(labels, t.sigmoid(output))
            auc_meter += auc_calc

        return loss_meter / it_count, f1_meter / it_count, auc_meter / it_count, acc_meter / it_count

    def _val_epoch(self):
        """
        Validate for an epoch by looping over the batches

        :return: val_loss, val_f1, val_auc
        """
        # set eval mode
        self._model.eval()
        # disable gradient computation
        f1_meter, loss_meter, auc_meter, acc_meter, it_count = 0, 0, 0, 0, 0
        with t.no_grad():
            # iterate through the validation set
            for images, labels in self._val_test_dl:
                if self._cuda:
                    # transfer the batch to the gpu if given
                    images = images.to(t.device("cuda"))
                    labels = labels.to(t.device("cuda"))
                # perform a validation step
                # predict
                output = self._model(images)
                # propagate through the network and calculate the loss and predictions
                loss = self._crit(output, labels)
                loss_meter += loss.item()
                it_count += 1
                f1 = self._calc_f1(labels, t.sigmoid(output))
                f1_meter += f1
                acc_meter = self._calc_acc(labels, t.sigmoid(output))
                acc_meter += acc_meter
                auc_calc = self._calc_auc(labels, t.sigmoid(output))
                auc_meter += auc_calc

        return loss_meter / it_count, f1_meter / it_count, auc_meter / it_count, acc_meter / it_count

    def train(self):
        """
        Main function for the training cycle.

        :return: best_f1
        """
        assert self._early_stopping_patience > 0 or config_params.max_epoch > 0
        # load model
        if self._args.ckpt:
            state = t.load(self._args.ckpt, map_location='cpu')
            self._model.load_state_dict(state['state_dict'])
            print('Train with pretrained weight val_f1', state['f1'])

        best_f1 = -1
        best_acc = -1
        lr = self._optim.param_groups[0]['lr']
        start_epoch = 1
        stage = 1
        min_val_loss = np.Inf

        save_dir = '%s/%s_%s' % (config_params.ckpt, config_params.model_name, time.strftime("%d_%m_%Y__%H_%M"))
        if self._args.ex:
            save_dir += self._args.ex

        logger = SummaryWriter(log_dir=save_dir)

        for self.epoch in range(start_epoch, config_params.max_epoch + 1):
            since = time.time()
            # stop by epoch number
            if self.epoch == config_params.max_epoch:
                break

            # train and validate for an epoch
            train_loss, train_f1, train_auc, train_acc = self._train_epoch()
            val_loss, val_f1, val_auc, val_acc = self._val_epoch()

            # report to tune
            if self._args.command == "hypertrain":
                tune.report(loss=val_loss, accuracy=val_f1)
            # print ergs
            print('#Epoch:%02d Stage:%d train_loss:%.3f train_f1:%.3f train_acc:%.3f train_auc:%.3f'
                  '\t val_loss:%0.3f val_f1:%.3f val_acc:%.3f val_auc:%.3f time:%s\n'
                  % (self.epoch, stage, train_loss, train_f1, train_acc, train_auc, val_loss, val_f1, val_acc, val_auc, self._time(since)))
            # add values to tensorboard logger
            logger.add_scalar('train/train_loss', train_loss, global_step=self.epoch)
            logger.add_scalar('train/train_f1', train_f1, global_step=self.epoch)
            logger.add_scalar('train/train_auc', train_auc, global_step=self.epoch)
            logger.add_scalar('train/val_acc', val_acc, global_step=self.epoch)
            logger.add_scalar('val/val_loss', val_loss, global_step=self.epoch)
            logger.add_scalar('val/val_f1', val_f1, global_step=self.epoch)
            logger.add_scalar('val/val_auc', val_auc, global_step=self.epoch)
            logger.add_scalar('val/val_acc', val_acc, global_step=self.epoch)
            state = {"state_dict": self._model.state_dict(), "epoch": self.epoch, "loss": val_loss, "f1": val_f1, "lr": lr,
                     "stage": stage}
            # save_checkpoint function to save the model
            self._save(state, best_acc < val_acc, save_dir)

            best_acc_before = best_acc
            best_acc = max(best_acc, val_acc)

            best_f1_before = best_f1
            best_f1 = max(best_f1, val_f1)

            # update lr
            if self.epoch in config_params.stage_epoch:
                stage += 1
                lr /= config_params.lr_decay
                best_weights = os.path.join(save_dir, config_params.best_w)
                self._model.load_state_dict(t.load(best_weights)['state_dict'])
                print("Step into Stage%02d lr %.3ef" % (stage, lr))
                self._adjust_lr(self._optim, lr)

            # Early stopping
            # If the validation loss is at a minimum
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1

                if round(best_f1, 2) > round(best_f1_before, 2):
                    # F1 increasing keep training
                    epochs_no_improve = 0

                if round(train_f1, 2) == 1:
                    print("Early stopping Train F1 == 1!")
                    logger.flush()
                    break

            if self.epoch > 10 and epochs_no_improve == self._early_stopping_patience:
                print('Early stopping!')
                logger.flush()
                break
            else:
                self.epoch += 1
                continue

        logger.flush()
        with tune.checkpoint_dir(self.epoch) as checkpoint_dir:
            path = os.path.join(save_dir, "checkpoint")

        return best_f1, best_acc

    def val(self, model_path):
        """
        Validate the model with the testing data.

        :param model_path: path to trained model
        :return: None
        """
        # load model to cpu
        self._model.load_state_dict(t.load(model_path, map_location='cpu')['state_dict'])
        # disable gradient computation
        self._model.eval()
        f1_meter, loss_meter, auc_meter, acc_meter, it_count = 0, 0, 0, 0, 0
        y_pre_list = np.array([])
        label_list = np.array([])
        with t.no_grad():
            # iterate through the validation set
            for images, labels in self._test_dl:
                if self._cuda:
                    # transfer the batch to the gpu if given
                    images = images.to(t.device("cuda"))
                    labels = labels.to(t.device("cuda"))
                # perform a validation step
                # predict
                output = self._model(images)
                # propagate through the network and calculate the loss and predictions
                loss = self._crit(output, labels)
                loss_meter += loss.item()
                it_count += 1
                output = t.sigmoid(output)
                # append predicted results of batch
                y_true, y_pre = self._list_build(labels, output, threshold=0.5)
                y_pre_list = np.concatenate([y_pre_list, y_pre])
                label_list = np.concatenate([label_list, y_true])
                # F1
                f1 = self._calc_f1(labels, output, threshold=0.5)
                f1_meter += f1
                # Accuracy
                accuracy = self._calc_acc(labels, output, threshold=0.5)
                acc_meter += accuracy
                # Auc
                auc_ = self._calc_auc(labels, output, threshold=0.5)
                auc_meter += auc_

        conf_matrix = self._confusion_matrix(y_true=label_list, y_pre=y_pre_list)
        f1_test = f1_meter / it_count
        auc_test = auc_meter / it_count
        acc_test = acc_meter / it_count
        print("Performance on Test Set:")
        print("Confusion Matrix: \n", conf_matrix)
        print("Test_F1:%.3f, Test_ACC:%.3f, Test_AUC:%.3f" % (f1_test, acc_test, auc_test))
        return conf_matrix, f1_test, auc_test, acc_test
