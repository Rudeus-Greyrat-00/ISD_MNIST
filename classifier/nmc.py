import numpy as np
from sklearn.metrics import pairwise_distances


class NMC(object):

    def __init__(self):
        self._centroids = None

    @property
    def centroids(self):
        return self._centroids

    def fit(self, xtr, ytr):
        num_classes = np.unique(ytr).size
        num_features = xtr.shape[1]  # the wideness of the xtr matrix
        centroids = np.zeros(
            shape=(num_classes, num_features))  # features == "dimension", so in this case width * heigh

        for k in range(num_classes):
            xk = xtr[ytr == k, :]  # if f.e. k = 0 xk will contains a lot of images of zeros
            centroids[k, :] = np.mean(xk,
                                      axis=0)  # we average along the specified axis (we average all the row so we get a single row averaged, then we put it in centroids[k, :])

        self._centroids = centroids
        # return centroids  # a matrix width as the image (width * heigh) and heigh as number of classes. Each row is an "averaged" image of a specific class

        return self  # by convenction

    def predict(self, xts):

        if self._centroids is None:
            raise ValueError("The classifier is not trained")

        dist = pairwise_distances(xts, self._centroids)
        y_pred = np.argmin(dist, axis=1)
        return y_pred
