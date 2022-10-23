import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from classifier import NMC
from data_loader import CDataLoaderMnist
from data_perturb import CDataPerturbRandom
from data_perturb import CDataPerturbGaussian


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
    plt.show()

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
data_loader = CDataLoaderMnist(filename='data/mnist_train_small.csv')
x, y = data_loader.load_data()

xtr, ytr, xts, yts = split_data(x, y, 0.6)

clf = NMC()
clf.fit(xtr, ytr)

Ks = [10, 20, 50, 100, 200, 500]
sigmas = [10, 20, 200, 200, 500]  # not sure why 200 appears 2 times. Maybe an error in the PDF?

Ker = []
sigmaer = []

R = CDataPerturbRandom()
G = CDataPerturbGaussian()

for K in Ks:
    R.K = K
    Xp = R.perturb_dataset(xts)
    y_pred = clf.predict(Xp)
    Ker.append(test_error(y_pred, yts))

for sigma in sigmas:
    G.sigma = sigma
    Xp = G.perturb_dataset(xts)
    y_pred = clf.predict(Xp)
    sigmaer.append(test_error(y_pred, yts))

# not sure about this part

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(Ks, Ker)
plt.title("Random noise for increasing K")
plt.subplot(1, 2, 2)
plt.plot(sigmas, sigmaer)
plt.title("Random noise for increasing sigma")
plt.show()

