#!/usr/bin/env python
__author__ = "Felix Tempel"
__credits__ = ["Felix Tempel"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Felix Tempel"
__email__ = "felixtempel95@hotmail.de"
__status__ = "Production"

import numpy as np
import skimage.transform
import copy


class Augmentation:

    def __init__(self):
        pass

    def execute(self, signal, dim, aug_prob, plot=False):
        """
        Execute the augmentation.

        :param plot: bool
        :param signal: np.array
        :param dim: tuple(int, int)
        :param aug_prob: float
        :return: np.array
        """
        if plot:
            self.plotter(signal, dim)

        # if np.random.uniform(0, 1) < aug_prob:
        #    choice = np.random.randint(0, 3)
        if np.random.uniform(0, 1) < aug_prob:
            # crop
            crop_choice = np.random.randint(low=0, high=8)
            signal = self._crop(signal, crop_choice, dim)
        if np.random.uniform(0, 1) < aug_prob:
            # mask square
            signal = self._mask_square(signal, dimension=np.random.randint(10, 30))
        # if np.random.uniform(0, 1) < aug_prob:
            # randomly rotate
        #    signal = self._random_rotate(signal, angle=np.random.uniform(-20, 20))

        return signal

    def _crop(self, signal, crop_choice, dimension):
        """
        Crop the signal into given directions.

        :param signal: np.array
        :param crop_choice: int
        :param dimension: int
        :return: np.array
        """
        if crop_choice == 0:
            # Left Top Crop
            crop = signal[:120, :120]
        elif crop_choice == 1:
            # Center Top Crop
            crop = signal[:120, 4:124]
        elif crop_choice == 2:
            # Right Top Crop
            crop = signal[:120, 8:]
        elif crop_choice == 3:
            # Left Center Crop
            crop = signal[4:124, :120]
        elif crop_choice == 4:
            # Center Center Crop
            crop = signal[4:124, 4:124]
        elif crop_choice == 5:
            # Right Center Crop
            crop = signal[4:124, 8:]
        elif crop_choice == 6:
            # Left Bottom Crop
            crop = signal[8:, :120]
        elif crop_choice == 7:
            # Center Bottom Crop
            crop = signal[8:, 4:124]
        elif crop_choice == 8:
            # Right Bottom Crop
            crop = signal[8:, 8:]

        cropped = skimage.transform.resize(crop, dimension, anti_aliasing=True)

        return cropped

    def _scaling(self, signal, sigma=0.1):
        """
        Scale the signal.

        :param signal: np.array
        :param sigma: float
        :return: np.array
        """
        scaling = np.random.normal(loc=1.0, scale=sigma, size=signal.shape[0])
        return signal * scaling

    def _vertical_flip(self, signal):
        """
        Flip the signal vertically.

        :param signal: np.array
        :return: np.array
        """
        return signal[:] * -1

    def _random_rotate(self, signal, angle):
        """
        Rotate the signal in a given angle
        :param signal:
        :param degrees:
        :return:
        """
        return skimage.transform.rotate(signal, angle, mode='constant', cval=1.0)

    def _mask_square(self, signal, dimension=15):
        """
        Generate a square mask which is put on the signal.

        :param signal: np.array
        :param dimension: int
        :return: np.array
        """
        # randomly generate coordinates
        x = np.random.randint(dimension, 128-dimension)
        y = np.random.randint(dimension, 128-dimension)
        signal[x:x+dimension, y:y+dimension] = 1
        return signal

    def _zero_burst(self, signal, threshold=2.5, depth=10):
        """
        Apply a random zero burst to the signal with given probability and depth.

        :param signal: np.array
        :param threshold: float
        :param depth: int
        :return: np.array
        """
        shape = signal.shape
        noise_shape = [1, shape[0] + depth]
        # Generate random noise
        noise = np.random.normal(0, 1, noise_shape)
        # Pick positions where the noise is above a certain threshold
        mask = np.greater(noise, threshold)
        # grow a neighbourhood of True values with at least length depth+1
        for d in range(depth):
            mask = np.logical_or(mask[:, :-1], mask[:, 1:])

        output = np.where(mask, np.zeros(shape), signal)
        return output[0]

    def plotter(self, signal, dim):
        """
        Plotter for documentation.

        :param signal: np.array
        :param dim: tuple(int, int)
        :return: None
        """
        import matplotlib
        matplotlib.use('MacOSX')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(4, 4, edgecolor='k')
        fig.tight_layout()
        axs = axs.ravel()

        for i in range(16):
            dummy = copy.deepcopy(signal)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            if i < 4:
                crop_choice = np.random.randint(low=0, high=8)
                sig = self._crop(dummy, i, dim)
                axs[i].imshow(sig, cmap="gray")
                axs[i].title.set_text('Crop')
            if 4 <= i < 8:
                # mask square
                sig = self._mask_square(dummy, dimension=15)
                axs[i].imshow(sig, cmap="gray")
                axs[i].title.set_text('Mask')
            if 8 <= i < 12:
                # randomly rotate
                sig = self._random_rotate(dummy, angle=np.random.uniform(-20, 20))
                axs[i].imshow(sig, cmap="gray")
                axs[i].title.set_text('Rotate')

            if i >= 12:
                # all
                crop_choice = np.random.randint(low=0, high=8)
                sig = self._crop(dummy, crop_choice , dim)
                sig = self._mask_square(sig, dimension=np.random.randint(5, 15))
                sig = self._random_rotate(sig, angle=np.random.uniform(-20, 20))
                axs[i].imshow(sig, cmap="gray")
                axs[i].title.set_text('Combination')
