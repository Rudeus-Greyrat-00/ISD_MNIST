from abc import ABC, abstractmethod

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
        pass
