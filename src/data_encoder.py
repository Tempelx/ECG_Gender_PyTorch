#!/usr/bin/env python
__author__ = "Felix Tempel"
__copyright__ = "Copyright 2020, ECG Sex Classification"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
import tensorboard as tb
import wfdb
from biosppy.signals import ecg


class DataEncoder:
    def __init__(self):
        # codes for normal rhythm
        self.snomed_ct_code_norm_2 = "426285000"
        self.snomed_ct_code_norm = "426783006"

        # Tensorboard experiments
        self.tb_chinese = "qXDQdR7PQbiKcWR8tj5few"
        self.tb_georgia = "DdDXpf3OTNqVZbqiPAX0OA"
        self.tb_ptb = "V7H4R5msQsaHKX0kBdvNug"
        self.tb_combination = "Bg2l9A6VSdSPLhYa7jPHMA"

        # MIT data
        self.df_mit = pd.read_csv("./src/ref/df_mit_op.csv")
        self.dir_mit = "/database/mit-bih-arrhythmia-database"
        self.records_mit = "/database/mit-bih-arrhythmia-database/RECORDS.txt"

    def find_number(self, text, c):
        return re.findall(r'%s(\d+\.\d+)' % c, text)

    def update_df_name(self, df):

        for index, row in df.iterrows():
            batch_size = re.findall(r'%s(\d+)' %"batch_size=", row.run)
            momoentum = self.find_number(row.run, "momentum=")
            lr = self.find_number(row.run, "lr=")
            str_name = "bs=" + str(batch_size)[2:-2]+ ", mom=" + str(momoentum)[2:-2] + ", lr=" + str(lr)[2:-5]
            df.loc[index, "run"] = str_name
        return df


    def tensorboard_to_csv(self):
        """
        Function to convert Tensorflow file to plots.

        :return: None
        """
        experiment = tb.data.experimental.ExperimentFromDev(self.tb_chinese)
        df = experiment.get_scalars()

        df_train_f1 = df[df.tag.str.endswith("train_f1")]
        df_train_loss = df[df.tag.str.endswith("train_loss")]

        df_val_f1 = df[df.tag.str.endswith("val_f1")]
        df_val_loss = df[df.tag.str.endswith("val_loss")]

        # update name
        df_train_f1 = self.update_df_name(df_train_f1)
        df_train_loss = self.update_df_name(df_train_loss)

        df_val_f1 = self.update_df_name(df_val_f1)
        df_val_loss = self.update_df_name(df_val_loss)

        # plot
        plt.figure(1)
        fig1 = sns.lineplot(data=df_train_f1, x="step", y="value", hue="run").set_title("Chinese Database F1 train set")
        plt.legend(fontsize=8)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim([0, 30])
        plt.xticks(np.arange(0, 30, 5))


        plt.figure(2)
        fig2 = sns.lineplot(data=df_train_loss, x="step", y="value", hue="run").set_title("Chinese Database Loss train set")
        plt.legend(fontsize=8)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim([0, 30])
        plt.xticks(np.arange(0, 30, 5))

        plt.figure(3)
        fig3 = sns.lineplot(data=df_val_f1, x="step", y="value", hue="run").set_title("Chinese Database Accuracy validation set")
        plt.legend(fontsize=8)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xlim([0, 30])
        plt.xticks(np.arange(0, 30, 5))

        plt.figure(4)
        fig4 = sns.lineplot(data=df_val_loss, x="step", y="value", hue="run").set_title("Chinese Database Loss validation set")
        plt.legend(fontsize=8)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        plt.xlim([0, 30])
        plt.xticks(np.arange(0, 25, 5))

        breakpoint()

    def execute_mit(self):
        """
        Get R-Peak positions from MIT database.

        :param dataframe: pd.df
        :param dir: str
        :return: None
        """
        # load dataframe

        df_mit = self._extract_mit()
        df_mit.to_csv('df_mit_op.csv', index=False)

        # self._mit_to_mat()

    def _mit_to_mat(self):
        """
        Convert MIT to mat file.

        :return: None
        """
        with open(self.records_mit,"r") as f:
            records = f.readlines()
        records = [x.strip() for x in records]

        for value in records:
            rec = value
            name = os.path.join(self.dir_mit, rec)
            data = wfdb.rdsamp(name)

            mat_dic = {"val": data[0][0:25000,:].T}
            rec = rec + ".mat"
            save_path = os.path.join(self.dir_mit, "mat", rec)

            scipy.io.savemat(save_path, mat_dic)

    def _extract_mit(self):
        """
        Helper function for mit extraction.

        :return: Dataframe
        """

        dataframe = pd.DataFrame(columns=["recording", "scp_codes", "sex", "age", "r_peak_pos", "r_peak_before",
                                   "r_peak_after", "lead"])

        with open(self.records_mit,"r") as f:
            records = f.readlines()
        records = [x.strip() for x in records]

        for value in records:
            rec = value
            name = os.path.join(self.dir_mit, rec)
            data = wfdb.rdsamp(name)

            age = data[1]["comments"][0][:2]
            sex = data[1]["comments"][0][3:4]
            if sex == "M":
                sex = 0
            elif sex == "F":
                sex = 1

            lead_1 = 0
            lead_2 = 1

            leads = [lead_1, lead_2]

            fs = data[1]["fs"]

            # loop over signal
            # whole sig??
            mit_ecg = ecg.ecg(data[0][0:25000, 1], sampling_rate=fs, show=False)
            dummy = pd.DataFrame(columns=["recording", "scp_codes", "sex", "age", "r_peak_pos", "r_peak_before",
                                          "r_peak_after", "lead"])
            for idx, peak in enumerate(mit_ecg["rpeaks"][:-1]):
                if idx == 0:
                    continue

                peak_before = mit_ecg['rpeaks'][idx-1]
                peak_after = mit_ecg['rpeaks'][idx+1]

                for lead in leads:
                    append_dic = {"recording": rec, "scp_codes": 1, "sex": sex, "age": age, "r_peak_pos": peak,
                                  "r_peak_before": peak_before,
                                  "r_peak_after": peak_after, "lead": lead}
                    dataframe = dataframe.append(append_dic, ignore_index=True)


        return dataframe

    def get_peaks(self, dataframe, dir):
        """
        Extract the peaks from the first Lead.

        :param dataframe: dataframe to append to
        :param dir: location
        :return: dataframe
        """
        df = pd.DataFrame(columns=["recording", "scp_codes", "sex", "age", "r_peak_pos", "r_peak_before",
                                   "r_peak_after", "lead"])
        for index in range(len(dataframe)):
            # load data
            path = os.path.join(dir, str(dataframe.recording[index]))
            path = path.replace('/', '\\')
            signal_raw = scipy.io.loadmat(path)

            ecg_object = ecg.ecg(signal=signal_raw['val'][0], sampling_rate=500, show=False)
            # TODO> detect motion artifacts and discard these peaks
            num_peaks = ecg_object['rpeaks'].size - 2
            peaks = ecg_object['rpeaks'][1:-1]
            peaks_before = ecg_object['rpeaks'][0:-2]
            peaks_after = ecg_object['rpeaks'][2:]
            dummy = pd.DataFrame(columns=["recording", "scp_codes", "sex", "age", "r_peak_pos", "r_peak_before",
                                          "r_peak_after", "lead"])
            dummy = dummy.append([dataframe.iloc[index]] * num_peaks, ignore_index=True)
            dummy.r_peak_pos = peaks
            dummy.r_peak_before = peaks_before
            dummy.r_peak_after = peaks_after

            for lead in range(0, 12):
                dummy.lead = lead
                df = df.append([dummy], ignore_index=True)

        return df

    def execute_single_peaks(self):
        """
        Execute single peaks.

        :return: None
        """

        df_chin_op = self.get_peaks(self.df_chin, self.dir)
        df_chin_op.to_csv('./ref/df_chin_op.csv', index=False)

        df_georgia_op = self.get_peaks(self.df_georgia, self.dir)
        df_georgia_op.to_csv('./ref/df_georgia_op.csv', index=False)

        df_ptb_op = self.get_peaks(self.df_ptb, self.dir)
        df_ptb_op.to_csv('./ref/df_ptb_op.csv', index=False)

        df_chin_2_op = self.get_peaks(self.df_chin_2, self.dir)
        df_chin_2_op.to_csv('./ref/df_chin_2_op.csv', index=False)

        print("Done")

    def get_df(self):
        """
        Execute for whole signal.
        sex (male 0, female 1)
        :return: None
        """

        # chinese
        dir_chin = ""
        df_chin = self._df_generator(dir_chin)
        df_chin.to_csv('./ref/df_chin.csv', index=False)

        # ptb
        dir_ptb = ""
        df_ptb = self._ptb_generator("")
        df_ptb.to_csv('./ref/df_ptb.csv', index=False)

        # georgia
        dir_georgia = ""
        df_georgia = self._df_generator(dir_georgia)
        df_georgia.to_csv('./ref/df_georgia.csv', index=False)

        # chin 2
        dir_chin_2 = ""
        df_chin_2 = self._df_generator(dir_chin_2)
        df_chin_2.to_csv('./ref/df_chin_2.csv', index=False)


        df_chin = pd.read_csv("/ref/df_chin.csv")
        df_chin_2 = pd.read_csv("/ref/df_chin_2.csv")
        df_ptb = pd.read_csv("/ref/df_ptb.csv")
        df_georgia = pd.read_csv("/ref/df_georgia.csv")

        max_chin, min_chin = self._max_lenght(dir_chin, df_chin)
        max_chin_2, min_chin_2 = self._max_lenght(dir_chin_2, df_chin_2)
        # max_ptb, min_ptb = self._max_lenght(dir_ptb, df_ptb)
        # max_georgia, min_georgia = self._max_lenght(dir_georgia, df_georgia)
        breakpoint()

    def _max_lenght(self, dir, df):
        """
        Get the max length of recording.

        :param dir: location
        :param df: dataframe
        :return: max & min length
        """
        max_length = 0
        min_length = 2000
        data_type = ".mat"
        for filename in df.recording:
            # load data
            path = os.path.join(dir, filename) + data_type
            dummy = scipy.io.loadmat(path)
            length = dummy['val'][0].size
            if length > max_length:
                max_length = length
            if length < min_length:
                min_length = length

        return max_length, min_length

    def _ptb_generator(self, dir):
        """
        Generate data for PTB database.

        :param dir: location
        :return: dataframe
        """
        df_dataset_ptb = pd.read_csv(dir)
        idx_drop = df_dataset_ptb.report[df_dataset_ptb.report != "sinusrhythmus normales ekg"]
        # norm 100 take 500 hz
        df_dataset_ptb = df_dataset_ptb.drop(df_dataset_ptb.index[idx_drop.index.values], axis=0).reset_index(
            drop=True)

        df = df_dataset_ptb[["filename_hr", "scp_codes", "sex", "age"]].copy()
        df.age = df.age.astype(int)
        df = df.rename(columns={"filename_hr": "recording"})
        df.recording = "HR" + df.recording.str[-8:-3]
        df.scp_codes = 1
        return df

    def _df_generator(self, dir):
        """
        Generate dataframe.

        :param dir: location
        :return: dataframe
        """
        dir_files = sorted(os.listdir(dir))
        df = pd.DataFrame(columns=["recording", "scp_codes", "sex", "age"])
        for filename in dir_files:
            if filename.endswith(".hea"):
                dummy = pd.read_csv(os.path.join(dir, filename), header=None, sep='\n')
                if dummy.iloc[15].item().__contains__(self.snomed_ct_code_norm) or dummy.iloc[15].item().__contains__(
                        self.snomed_ct_code_norm_2):
                    age = dummy.iloc[13].item().split(" ")[1]
                    gender = dummy.iloc[14].item().split(" ")[1]
                    if gender == "Female":
                        gender = 1
                    elif gender == "Male":
                        gender = 0
                    df = df.append({"recording": filename[:-4], "scp_codes": 1, "sex": gender, "age": age}, ignore_index=True)
            else:
                continue
        return df


if __name__ == '__main__':
    pass
    # encoder = DataEncoder()
    # encoder.execute_mit()
    # encoder.tensorboard_to_csv()
