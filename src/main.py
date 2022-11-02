import numpy as np
import matplotlib.pyplot as plt
from src.classifier import NMC
from src.data_loader import CDataLoaderMnist
from src.data_perturb import CDataPerturbRandom
from src.data_perturb import CDataPerturbGaussian


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

def test_accuracy(y_pred, yts):
    return (y_pred == yts).mean()

def classifiy_perturb_data_err(clf, xts, yts, perturb_v, perturbator, attr):
    error_rate = []
    if not hasattr(perturbator, attr):
        raise ValueError("Perturbator has not a parameter named ", attr)
    for perturb in perturb_v:
        setattr(perturbator, attr, perturb)
        Xp = perturbator.perturb_dataset(xts)
        y_pred = clf.predict(Xp)
        error_rate.append(test_error(y_pred, yts))
    return error_rate

def classifiy_perturb_data_acc(clf, xts, yts, perturb_v, perturbator, attr):
    acc_rate = []
    if not hasattr(perturbator, attr):
        raise ValueError("Perturbator has not a parameter named ", attr)
    for perturb in perturb_v:
        setattr(perturbator, attr, perturb)
        Xp = perturbator.perturb_dataset(xts)
        y_pred = clf.predict(Xp)
        acc_rate.append(test_accuracy(y_pred, yts))
    return acc_rate


# START
data_loader = CDataLoaderMnist(filename='data/mnist_train_small.csv')
x, y = data_loader.load_data()

xtr, ytr, xts, yts = split_data(x, y, 0.6)

clf = NMC()
clf.fit(xtr, ytr)

Ks = [0, 10, 20, 50, 100, 200]
#sigmas = [10, 20, 200, 200, 500]  # not sure why 200 appears 2 times. Maybe an error in the PDF?
# is it supposed to be like this --> sigmas = Ks ?
sigmas = Ks

Ker = []
sigmaer = []

R = CDataPerturbRandom()
G = CDataPerturbGaussian()

Ker = classifiy_perturb_data_acc(clf, xts, yts, Ks, R, "K")
sigmaer = classifiy_perturb_data_acc(clf, xts, yts, Ks, G, "sigma")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(Ks, Ker)
plt.title("Accuracy on random noise for increasing K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.subplot(1, 2, 2)
plt.plot(sigmas, sigmaer)
plt.title("Accuracy on gaussian noise for increasing sigma")
plt.xlabel("sigma")
plt.ylabel("Accuracy")
plt.show()

