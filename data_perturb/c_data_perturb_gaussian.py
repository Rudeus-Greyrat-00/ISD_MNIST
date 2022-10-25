from .c_data_perturb import CDataPerturb
import numpy as np


class CDataPerturbGaussian(CDataPerturb):

    def __init__(self, min_value=0, max_value=1, sigma=100.0):
        self._min_value = min_value
        self._max_value = max_value
        self._sigma = sigma

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def sigma(self):
        return self._sigma

    @min_value.setter
    def min_value(self, value):
        if value < 0 or value > 1:
            raise ValueError("Min value parameter should be within 0 and 1")
        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        if value < 0 or value > 1 or value <= self._min_value:
            raise ValueError("Max value parameter should be within 0 and 1 and greater than min value")
        self._max_value = value

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def data_perturbation(self, x):
        """
        Gaussian perturbation
        :param x:
        :return:
        """
        xp = x.copy()
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal
        xp[:] += np.random.normal(loc=0, scale=self._sigma, size=xp.shape[0]) / 255
        # we divide by 255 because we used the range 0 - 1 to store the values of the images
        # so without dividing by 255, the perturbation with the default value of 100 for sigma that would work for
        # a range of 0 - 255 for storing images values, would be excessive.
        xp[xp < self._min_value] = self._min_value
        xp[xp > self._max_value] = self._max_value
        return xp
