#!/usr/bin/env python
__author__ = "Felix Tempel"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal
from biosppy.signals import ecg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import lfilter
from skimage.color import rgb2gray, rgba2rgb
from sklearn import preprocessing

from src.config import config_params


class GenderLoader:

    def __init__(self, train, win=6, hop=3, nfft=12, plot=False):
        # signal parameters
        self.f_s = 500
        self.window = int((self.f_s * 0.25) / 2)

        self.train = train

        # spectrogram parameters
        self.cmap = plt.get_cmap('gray')
        self.mode = 'magnitude'
        # width of the hanning window used for the spectrogram
        self.win = win
        # spectrogram hanning window overlap
        self.hop = hop
        self.nfft = nfft
        self.plot_mode = plot

    def execute_peak(self, signal_raw, lead, r_peak, before, after):
        """
        Generate a grayscale image from the signal at the given R-peak position.

        :param signal_raw: numpy array of unprocessed signal
        :param lead: lead to be executed
        :param r_peak: location of peak
        :param before: location of peak before
        :param after: location of the peak coming after
        :return: np.float32 grayscale image
        """

        if self.plot_mode:
            self.debug(signal_raw[lead], lead, r_peak, before, after)

        # remove Baseline
        signal = self._baseline_remove(signal_raw[lead])
        # convolve
        # signal = convolve(signal, kernel=Box1DKernel(7.5))
        signal = scipy.signal.savgol_filter(signal, 13, 2)
        # normalize
        signal = preprocessing.minmax_scale(signal, feature_range=(0, 1))

        # get area
        diff1 = abs(before - r_peak)
        diff2 = abs(after - r_peak)
        x = r_peak - diff1 // 2
        y = r_peak + diff2 // 2
        signal = signal[x:y]

        fig = plt.figure(figsize=config_params.dimension, dpi=100)
        fig.add_axes([0., 0., 1., 1.])
        canvas = FigureCanvasAgg(fig)
        plt.plot(signal, 'k')
        plt.axis('off')
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close(fig)
        sample = np.asarray(buf)
        sample = rgb2gray(rgba2rgb(sample)).astype(np.float32)

        return sample

    def execute_spectro(self, signal_raw, lead, r_peak, before, after):
        """
        Generate a grayscale spectrogram image from the signal at the given R-peak position.

        :param signal_raw: numpy array of unprocessed signal
        :param lead: lead to be executed
        :param r_peak: location of peak
        :param before: location of peak before
        :param after: location of the peak coming after
        :return: np.float32 grayscale image
        """

        if self.plot_mode:
            self.debug(signal_raw[lead], lead, r_peak, before, after)

        # remove Baseline
        signal = self._baseline_remove(signal_raw[lead])
        # convolve
        # signal = convolve(signal, kernel=Box1DKernel(7.5))
        signal = scipy.signal.savgol_filter(signal, 13, 2)

        signal = preprocessing.minmax_scale(signal, feature_range=(0, 1))

        f, t, Sxx = self._spectrogram(signal, True)
        # to image
        fig = plt.figure(figsize=config_params.dimension, dpi=100)
        fig.add_axes([0., 0., 1., 1.])
        canvas = FigureCanvasAgg(fig)
        plt.pcolormesh(t, f, Sxx, edgecolors=None, shading='auto')
        plt.axis('off')
        canvas.draw()
        buf = canvas.buffer_rgba()
        plt.close(fig)
        sample = np.asarray(buf)
        sample = rgb2gray(rgba2rgb(sample)).astype(np.float32)

        return sample

    def _spectrogram(self, signal, log):
        """
        Transforms the input data to spectrogram

        :param signal: array of signal
        :param log: boolean
        :return: Spectrogram values
        """
        f, t, Sxx = scipy.signal.spectrogram(signal, fs=self.f_s, nperseg=self.win, noverlap=self.hop, nfft=self.nfft)
        if log:
            # mask values equal to 0
            Sxx = abs(Sxx)
            mask = Sxx > 0
            Sxx[mask] = 10 * np.log(Sxx[mask])
        return f, t, Sxx

    def _baseline_remove(self, signal, window=0.2, window_2=0.6):
        """
        Baseline remover with two median filters.

        :param signal: array of signal
        :param window: first window 20ms
        :param window_2: second window 60ms
        :return:
        """
        win_size = int(round(window * self.f_s))
        if win_size % 2 == 0:
            win_size += 1
        baseline_estimation = scipy.signal.medfilt(signal, kernel_size=win_size)
        win_size = int(round(window_2 * self.f_s))
        if win_size % 2 == 0:
            win_size += 1
        baseline_estimation = scipy.signal.medfilt(baseline_estimation, kernel_size=win_size)
        signal_baseline_free = signal - baseline_estimation

        return signal_baseline_free

    def butter_lowpass(self, cutoff, fs, order=5):
        """
        Butter lowpass filter.

        :param cutoff: int
        :param fs: int
        :param order: int
        :return: filter coeff
        """
        nyq = 0.5 * fs
        cutoff = cutoff / nyq
        b, a = scipy.signal.butter(order, cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, fwbw, order=5):
        """
        Apply a lowpass filter to the data.

        :param data: signal array
        :param cutoff: float frequency
        :param fs: sampling frequency
        :param fwbw: forward/backward filter
        :param order: int order of filter
        :return: filtered signal
        """
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        if fwbw:
            y = scipy.signal.filtfilt(b, a, data)
        else:
            y = scipy.signal.lfilter(b, a, data)
        return y

    def butter_bandpass(self, low_cut, high_cut, fs, order=5):
        """
        Apply a bandpass filter to the data.

        :param data: signal array
        :param cutoff: float frequency
        :param fs: sampling frequency
        :param fwbw: forward/backward filter
        :param order: int order of filter
        :return: filtered signal
        """
        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a bandpass filter to the data.

        :param data: signal array
        :param lowcut: float
        :param highcut: float
        :param fs: float sampling frequency
        :param order: int order of filter
        :return: filtered signal
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def debug(self, signal_raw, lead, r_peak, before, after):
        """
        Debug mode for image generation.

        :param signal_raw: numpy array of unprocessed signal
        :param lead: lead to be executed
        :param r_peak: location of peak
        :param before: location of peak before
        :param after: location of the peak coming after
        :return: None
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('macosx')

        T_sig = signal_raw.shape[0] / self.f_s
        t_vec = np.arange(0, T_sig, 1 / self.f_s)

        dummy = ecg.ecg(signal_raw, sampling_rate=self.f_s, show=False)
        plt.figure()
        plt.plot(dummy["filtered"], label="Signal")
        plt.plot(dummy["rpeaks"], dummy["filtered"][dummy["rpeaks"]], "*r", label="Detected R-Peaks")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        # plt.savefig("whole_sig.pdf")

        signal = self._baseline_remove(signal_raw)
        signal = scipy.signal.savgol_filter(signal, 13, 2)

        diff1 = abs(before - r_peak)
        diff2 = abs(after - r_peak)
        x = r_peak - diff1 // 2
        y = r_peak + diff2 // 2
        signal = signal[x:y]

        f, t, Sxx = self._spectrogram(signal, True)
        fig = plt.figure(figsize=config_params.dimension, dpi=100)
        fig.add_axes([0., 0., 1., 1.])
        plt.pcolormesh(t, f, Sxx, edgecolors=None, cmap=cm.gray, shading='auto')
        canvas = FigureCanvasAgg(fig)
        # plt.plot(signal, 'k')
        plt.axis('off')
