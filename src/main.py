import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from classifier import NMC


def load_mnist_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)  # convert to np.array

    # print(data.shape) row = images, first col = labels
    # X = [N*D][Y] see slide

    y = data[:, 0]  # all the row, only col 0. These are the LABELS. Labels goes from 0 to 9 and there are a lot of it
    x = data[:,
        1:] / 255  # the rest of the matrix, the image matrix (0 to 255 values for each pixels, we divide by 255 so the range is from 0 to 1)

    # the image are stored as row. Every image is 28 by 28 so as a row they become a vector wich lenght is 784

    return x, y


def plot_digit(image, shape=(28, 28)):
    plt.imshow(np.reshape(image, newshape=shape),
               cmap='gray')  # this would be the first image (each row is an image 28 by 28 rowed)


def plot_ten_digit(x, y=None):
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i + 1)  # i + 1 because subplots start from 1
        plot_digit(x[i, :])  # call our plot image function
        if y is not None:
            plt.title("Label: " + str(y[i]))


def split_data(x, y, tr_fraction=0.5):
    num_samples = y.size
    num_tr = int(tr_fraction * y.size)  # training
    num_ts = num_samples - num_tr

    tr_idx = np.zeros(num_samples, )  # [0, 0, 0, ..... 0]
    tr_idx[0:num_tr] = 1  # this are the tr index
    np.random.shuffle(tr_idx)  # randomize the vector

    ytr = y[tr_idx == 1]  # this works in python, pass to ytr all the index of tr_idx that are equals to 1
    yts = y[tr_idx == 0]

    xtr = x[tr_idx == 1, :]  # these row, always all colums
    xts = x[tr_idx == 0, :]

    return xtr, ytr, xts, yts

def test_error(y_pred, yts):
    return (y_pred != yts).mean()


# START
filename = 'data/mnist_train_small.csv'
clf = NMC()

# IMPLEMENT OF FULL PIPELINE

x, y = load_mnist_data(filename)

n_rep = 10
ts_err = np.zeros(shape=(n_rep,))

for rep in range(n_rep):
    xtr, ytr, xts, yts = split_data(x, y)
    clf.fit(xtr, ytr)
    y_pred = clf.predict(xts)
    ts_err[rep] = test_error(y_pred, yts)

print(ts_err.mean(), 2 * ts_err.std())  # standard deviation --> .std()
