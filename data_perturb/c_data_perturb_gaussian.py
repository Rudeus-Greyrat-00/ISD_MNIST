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
        if value < 0 and value > 1:
            raise ValueError("Min value parameter should be within 0 and 1")
        self._min_value = value

    @max_value.setter
    def max_value(self, value):
        if value < 0 and value > 1 and value >= self._min_value:
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
        pass


