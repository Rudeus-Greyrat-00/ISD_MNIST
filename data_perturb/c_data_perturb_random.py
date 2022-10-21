from c_data_perturb import CDataPerturb

class CDataPerturbRandom(CDataPerturb):

    def __init__(self, min_value=0, max_value=1, K=100):
        self._min_value = min_value
        self._max_value = max_value
        self._K = K

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def K(self):
        return self._K

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

    @K.setter
    def K(self, value):
        if K < 0:
            raise ValueError("K should be positive")
        self._K = value

    def data_perturbation(self, x):
        v = np.zeros(shape=x.shape[0])
        xp = np.zeros(shape=x.shape[0])
        if K > x.shape[0]:
            v = np.ones(shape=x.shape[0])
        else:
            for i in range(K):
                v[i] = 1
            np.random.shuffle(v)

        for item in range(v.shape[0]):
            if item == 0:
                xp[item] = x[item]
            else:
                xp[item] = np.random.uniform(self._min_value, self.max_value)













