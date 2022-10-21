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
        for i in range(X.shape[1]):
            self.data_perturbation(X[i, :])


