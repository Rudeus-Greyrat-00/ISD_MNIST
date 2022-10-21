from data_loader import CDataLoader
import pandas as pd
import numpy as np


class CDataLoaderMnist(CDataLoader):
    """
    Loader for MNIST handwritten digit
    """

    def __init__(self, filename="../data/mnist_train_small.csv"):
        self.filename = filename
        self._height = 28
        self._width = 28

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, str):
            raise ValueError("Filename parameter is not a string")
        else:
            self._filename = value

    def load_data(self):  # return x, y
        data = pd.read_csv(self.filename)
        data = np.array(data)  # convert to np.array

        # print(data.shape) row = images, first col = labels
        # X = [N*D][Y] see slide

        y = data[:,
            0]  # all the row, only col 0. These are the LABELS. Labels go from 0 to 9 and there are a lot of it
        x = data[:,
            1:] / 255  # the rest of the matrix, the image matrix (0 to 255 values for each pixel, we divide by 255
        # so the range is from 0 to 1)

        # the images are stored as row. Every image is 28 by 28 so as a row they become a vector which length is 784

        return x, y
