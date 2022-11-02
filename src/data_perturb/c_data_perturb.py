from abc import ABC, abstractmethod
import numpy as np


class CDataPerturb(ABC):
    """
    General interface as abstract class
    """

    def __init__(self):
        pass

    @abstractmethod
    def data_perturbation(self, x):
        pass

    def perturb_dataset(self, X):
        Xp = np.zeros(shape=X.shape)
        for i in range(X.shape[0]):
            Xp[i, :] = self.data_perturbation(X[i, :])

        return Xp
